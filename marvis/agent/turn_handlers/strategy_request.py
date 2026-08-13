"""strategy_request driver-turn handlers (executed into the package namespace by __init__.py)."""
from __future__ import annotations
from collections.abc import Mapping, Sequence
from pathlib import Path
import sqlite3
from marvis.agent.instruction_router import route_instruction
from marvis.agent.plan_driver import CONFIRMATION_SOURCE_AUTO, DriverError, is_confirm
from marvis.agent.semantic_intent import INTENT_STRATEGY_SAMPLE_BINDING
from marvis.agent.strategy_setup import STRATEGY_INTENT_FULL_DEVELOPMENT, STRATEGY_INTENT_MONITORING, STRATEGY_INTENT_RULE_MINING, StrategySetupError, build_strategy_dataset_context, preview_strategy_dataset_context
from marvis.agent.strategy_request_compiler import CompiledStrategyRequestDraft, StandardWorkflowRequestDraft, StrategyRequestDraft, compile_strategy_request, utterance_targets_candidate_monthly_stability, utterance_targets_interactive_tree_frontier_group_materialization, utterance_targets_model_score_comparison_v2, utterance_targets_scorecard_band_build, utterance_targets_scorecard_cutoff_selection, utterance_targets_strategy_dsl_delivery, utterance_targets_strategy_impact_cube, utterance_targets_strategy_pool_materialize, utterance_targets_strategy_pool_stability, utterance_targets_strategy_project_context, utterance_targets_strategy_report_bundle_v2, utterance_targets_strategy_sample_design, validate_strategy_request
from marvis.agent.strategy_workflows import StrategyWorkflowPreparationContext, StrategyWorkflowValidationError, prepare_strategy_plan
from marvis.agent.workflow_recovery import is_explicit_workflow_retry, latest_unresolved_workflow_failure
from marvis.data.errors import DatasetContentDriftError
from marvis.data.workspace import DataSemanticMapping, DataWorkspaceDraft
from marvis.repositories.strategy import StrategyRepository
from marvis.repositories.tasks import TaskRepository
from marvis.domain import TASK_TYPE_MODELING, TASK_TYPE_STRATEGY, TaskRecord
from marvis.llm_client import LLMClientError
from marvis.packs.strategy.project_context import strategy_project_context_structured_request_sha256
from marvis.repositories.pending_strategy_requests import PendingStrategyRequestConflictError, PendingStrategyRequestNotFoundError, PendingStrategyRequestRepository
from marvis.repositories.strategy_project_context import StrategyProjectContextDataError, StrategyProjectContextRepository
from marvis.repositories.data_workspace import DataWorkspaceDataError, DataWorkspaceDatasetNotFound, DataWorkspaceRepository, DataWorkspaceRevisionConflict

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # names defined by sibling lanes; merged into one namespace at runtime
    from . import DriverTurnRuntime
    from . import _MANUAL_STRATEGY_WORKFLOWS
    from . import _MODELING_INTAKE_PARAM_NAMES
    from . import _PROJECT_CONTEXT_ALL_PENDING_RE
    from . import _PROJECT_CONTEXT_ANSWER_PATTERNS
    from . import _PROJECT_CONTEXT_UNAVAILABLE_ANSWER_RE
    from . import _SPECIALIZED_WORKFLOW_INTAKE_TYPES
    from . import _STORED_EVALUATION_OPERATIONS
    from . import _STRATEGY_AUTOMATIC_TREE_SHORTHAND_RE
    from . import _STRATEGY_DROP_NAN_CANCEL_RE
    from . import _STRATEGY_DROP_NAN_CONFIRM_RE
    from . import _STRATEGY_MODEL_EVIDENCE_V2_REQUEST_RE
    from . import _STRATEGY_POOL_COMPILE_REQUEST_RE
    from . import _STRATEGY_POOL_IMPACT_REQUEST_RE
    from . import _STRATEGY_POOL_MEASUREMENT_WORKFLOWS
    from . import _STRATEGY_POOL_VALIDATION_REQUEST_RE
    from . import _STRATEGY_POOL_WORKFLOWS
    from . import _STRATEGY_REQUEST_ACTION_RE
    from . import _STRATEGY_REQUEST_CANCEL_RE
    from . import _STRATEGY_REQUEST_META_KEY
    from . import _STRATEGY_REQUEST_NON_EXECUTION_RE
    from . import _STRATEGY_REQUEST_SUBJECT_RE
    from . import _STRATEGY_SAMPLE_BOUND_CANDIDATE_WORKFLOWS
    from . import _STRATEGY_SAMPLE_DESIGN_V2_MISSING_CONTROLS
    from . import _StrategySampleDesignRequiredError
    from . import _StrategyV2EvidenceSetupError
    from . import _active_plan
    from . import _automatic_tree_apply_slots
    from . import _automatic_tree_leaf_materialization_slots
    from . import _bind_candidate_monthly_stability_evidence
    from . import _bind_candidate_source_artifact_evidence
    from . import _bind_scorecard_band_evidence
    from . import _bind_scorecard_cutoff_evidence
    from . import _bind_scorecard_model_score_evidence
    from . import _bind_univariate_dataset_evidence
    from . import _candidate_strategy_slots
    from . import _confirm_manual_sample_design_time_semantics
    from . import _cross_matrix_cell_selection_slots
    from . import _driver
    from . import _ensure_automatic_tree_active_workspace
    from . import _inherit_strategy_sample_drop_nan_policy
    from . import _instruction_explicitly_names_identifier
    from . import _interactive_tree_auto_continuation_plan_slots
    from . import _interactive_tree_revision_plan_slots
    from . import _interactive_tree_split_search_plan_slots
    from . import _is_auto_candidate_draft
    from . import _is_automatic_tree_build_draft
    from . import _latest_adhoc_pending
    from . import _latest_c1_state
    from . import _latest_feature_target_state
    from . import _latest_matching_strategy_sample_design_ref
    from . import _latest_strategy_nan_label_confirmation
    from . import _model_score_comparison_plan_slots
    from . import _modeling_data_runtime
    from . import _modeling_intake_param_schema
    from . import _platform_evidence_only
    from . import _repeat_strategy_nan_label_clarification
    from . import _resume_strategy_after_nan_label_confirmation
    from . import _run_strategy_setup
    from . import _semantic_intent_clarification_response
    from . import _semantic_strategy_sample_binding_context
    from . import _semantic_workflow_source_materials
    from . import _stored_strategy_slots
    from . import _strategy_contract_from_draft
    from . import _strategy_cross_candidate_build_from_search_plan_slots
    from . import _strategy_cross_candidate_search_plan_slots
    from . import _strategy_cross_rule_candidate_build_plan_slots
    from . import _strategy_cross_rule_search_plan_slots
    from . import _strategy_dataset_binding_matches
    from . import _strategy_dataset_context
    from . import _strategy_dataset_preview
    from . import _strategy_dsl_delivery_plan_slots
    from . import _strategy_impact_cube_dataset_preview
    from . import _strategy_impact_cube_plan_slots
    from . import _strategy_model_evidence_v2_plan_slots
    from . import _strategy_nan_label_clarification_response
    from . import _strategy_pool_apply_plan_slots
    from . import _strategy_pool_impact_dataset_context
    from . import _strategy_pool_impact_dataset_preview
    from . import _strategy_pool_impact_plan_slots
    from . import _strategy_pool_materialize_plan_slots
    from . import _strategy_pool_plan_slots
    from . import _strategy_pool_stability_plan_slots
    from . import _strategy_pool_validation_plan_slots
    from . import _strategy_project_context_plan_slots
    from . import _strategy_report_bundle_v2_plan_slots
    from . import _strategy_request_allowed_columns
    from . import _strategy_request_clarification_response
    from . import _strategy_request_preflight
    from . import _strategy_request_requires_complete_labels
    from . import _strategy_request_requires_dataset
    from . import _strategy_request_requires_target
    from . import _strategy_request_success_criteria
    from . import _strategy_sample_design_dataset_context
    from . import _strategy_sample_design_dataset_preview
    from . import _strategy_sample_design_plan_slots
    from . import _strategy_sample_design_v2_plan_slots
    from . import _strategy_slots_with_drop_nan
    from . import _strategy_target_nan_stats
    from . import _strategy_voting_candidate_build_from_search_plan_slots
    from . import _strategy_voting_candidate_plan_slots
    from . import _strategy_voting_candidate_search_plan_slots
    from . import _typed_strategy_slots
    from . import _validate_strategy_sample_design_target
    from . import append_driver_messages
    from . import append_join_error
    from . import join_turn_response
    from . import latest_open_gate

def _handle_strategy_sample_binding_intent(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    *,
    user_text: str,
) -> dict:
    """Bind the sole authenticated strategy sample without creating a plan.

    Two independent semantic passes authorize only the operation kind.  This
    handler never infers identifiers from action keywords: it grounds the exact
    file and target names in the utterance, re-resolves the single candidates
    exposed in the reviewed route context, validates the full target column,
    pins the normalized bytes, and atomically commits dataset target semantics,
    task target, workspace, audit, and conversation state.
    """

    try:
        route_context = _semantic_strategy_sample_binding_context(runtime, task)
        if not bool(route_context.get("strategy_sample_binding_available")):
            raise StrategySetupError(
                "当前材料不能唯一确定一个可绑定的策略样本与二元目标列。"
            )
        names = route_context.get("available_strategy_samples")
        if not isinstance(names, list) or len(names) != 1:
            raise StrategySetupError("当前策略样本候选不唯一。")
        expected_relative_name = str(names[0])
        expected_name = Path(expected_relative_name).name
        target_col = str(
            route_context.get("strategy_sample_target_candidate") or ""
        ).strip()
        if not target_col:
            raise StrategySetupError("当前策略样本的二元目标列不唯一。")
        if not _instruction_explicitly_names_identifier(user_text, expected_name):
            raise StrategySetupError(
                f"原话没有明确指定当前唯一策略样本 `{expected_name}`。"
            )
        if not _instruction_explicitly_names_identifier(user_text, target_col):
            raise StrategySetupError(
                f"原话没有明确指定当前目标/坏样本字段 `{target_col}`。"
            )

        source_materials = _semantic_workflow_source_materials(task)
        if len(source_materials) != 1 or str(
            source_materials[0].get("relative_path") or ""
        ) != expected_relative_name:
            raise StrategySetupError("策略样本材料在语义复核后发生变化。")
        expected_source_sha = str(source_materials[0].get("sha256") or "")

        workspace_repo = DataWorkspaceRepository(runtime.settings.db_path)
        snapshot = workspace_repo.get_or_default(task.id)
        if snapshot.active_dataset_id is not None:
            raise StrategySetupError(
                "当前 DataWorkspace 已有活动数据，未覆盖原绑定。"
            )

        backend, registry = _modeling_data_runtime(runtime.settings)
        preview = preview_strategy_dataset_context(
            registry,
            backend,
            task.id,
            task.source_dir,
            target_col=None,
        )
        if (
            preview.dataset_name != expected_name
            or preview.target_col != target_col
        ):
            raise StrategySetupError(
                "策略样本或目标列在绑定前发生变化。"
            )

        context = build_strategy_dataset_context(
            registry,
            backend,
            task.id,
            task.source_dir,
            target_col=target_col,
            require_target=True,
        )
        if (
            context.target_col != target_col
            or tuple(context.columns) != tuple(preview.columns)
            or not context.dataset_content_hash
        ):
            raise StrategySetupError(
                "注册后的策略样本身份或目标列与已复核候选不一致。"
            )
        source_identity = registry.source_identity(context.dataset_id)
        if (
            not isinstance(source_identity, dict)
            or source_identity.get("original_name") != expected_name
            or (
                expected_source_sha
                and source_identity.get("sha256") != expected_source_sha
            )
        ):
            raise StrategySetupError(
                "注册后的策略样本来源身份与已复核材料不一致。"
            )

        binding = registry.authenticate_dataset_binding(
            context.dataset_id,
            expected_task_id=task.id,
            expected_content_hash=context.dataset_content_hash,
        )
        if target_col not in set(
            registry.authenticated_binding_column_names(binding)
        ):
            raise StrategySetupError(
                f"目标列 `{target_col}` 不在认证后的策略样本中。"
            )
        _validate_strategy_sample_design_target(
            registry,
            backend,
            dataset_id=binding.dataset_id,
            target_col=target_col,
        )

        completion_content = (
            f"已绑定 `{expected_name}` 为当前策略样本，并将 `{target_col}` "
            "确认为目标/坏样本字段。当前只完成了受认证的数据绑定，"
            "没有创建或执行策略计划。\n\n"
            "要继续创建 StrategySampleDesign V2，请补充并确认："
            "坏样本取值（0/1）、缺失标签处理政策、审批人群与风险人群及其关系、"
            "分区规则、成熟度、表现窗、观察窗和字段绑定。"
            "平台不会替你推断这些口径。"
        )

        def persist_binding(conn: sqlite3.Connection) -> None:
            registry.persist_authenticated_target_on_connection(
                conn,
                binding,
                target_col,
            )
            repo.update_target_col_on_connection(conn, task.id, target_col)
            repo.add_agent_message_on_connection(
                conn,
                task.id,
                role="user",
                stage="chat",
                content=user_text,
                metadata={"intent": INTENT_STRATEGY_SAMPLE_BINDING},
            )
            repo.add_agent_message_on_connection(
                conn,
                task.id,
                role="assistant",
                stage="chat",
                content=completion_content,
                metadata={
                    "intent": INTENT_STRATEGY_SAMPLE_BINDING,
                    "kind": "data_workspace_binding",
                    "code": "strategy_sample_binding_complete",
                    "dataset_id": binding.dataset_id,
                    "dataset_content_hash": binding.content_hash,
                    "dataset_name": expected_name,
                    "dataset_relative_path": expected_relative_name,
                    "target_col": target_col,
                    "missing_controls": list(
                        _STRATEGY_SAMPLE_DESIGN_V2_MISSING_CONTROLS
                    ),
                },
            )

        workspace_repo.save_initial_binding(
            task.id,
            DataWorkspaceDraft(
                active_dataset_id=binding.dataset_id,
                active_dataset_content_hash=binding.content_hash,
                page="overview",
                selected_field=target_col,
                semantic_mapping=DataSemanticMapping(
                    target_col=target_col,
                    field_roles={target_col: "target"},
                    business_names={},
                ),
            ),
            expected_revision=snapshot.revision,
            audit={
                "actor": "user:strategy-sample-binding",
                "detail": {
                    "reason": (
                        "bind semantic-authorized strategy sample and target"
                    ),
                    "dataset_id": binding.dataset_id,
                    "dataset_content_hash": binding.content_hash,
                    "dataset_name": expected_name,
                    "dataset_relative_path": expected_relative_name,
                    "target_col": target_col,
                },
            },
            dataset_authenticator=lambda: (
                registry.verify_authenticated_binding_snapshot(binding)
            ),
            on_connection=persist_binding,
        )
        return join_turn_response(repo, task.id)
    except (
        DataWorkspaceDataError,
        DataWorkspaceDatasetNotFound,
        DataWorkspaceRevisionConflict,
        DatasetContentDriftError,
        KeyError,
        OSError,
        StrategySetupError,
        TypeError,
        ValueError,
    ) as exc:
        return _semantic_intent_clarification_response(
            repo,
            task,
            user_text=user_text,
            reason=str(exc),
        )

def _specialized_workflow_intake_is_pending(
    runtime: DriverTurnRuntime,
    task: TaskRecord,
    conversation: list[dict],
) -> bool:
    """Whether the task-owned workflow still owns its initial text turn.

    Task type is already a governed workflow selection.  Before that workflow
    has published a plan or a C1 setup contract, a generic data-analysis route
    must not reinterpret its first natural-language specification.  A previous
    fail-closed clarification does not consume the intake, so the user can
    safely retry with a clearer instruction.
    """

    if task.task_type not in _SPECIALIZED_WORKFLOW_INTAKE_TYPES:
        return False
    if runtime.plan_repo.list_plans_for_task(task.id):
        return False
    if _latest_c1_state(conversation) is not None:
        return False
    if _latest_feature_target_state(conversation) is not None:
        return False
    if _latest_adhoc_pending(conversation) is not None:
        return False
    return True

def _specialized_workflow_intake_route(
    runtime: DriverTurnRuntime,
    task: TaskRecord,
    *,
    instruction: str,
) -> tuple[dict[str, object] | None, str]:
    """Ask the selected workflow's own LLM contract whether intake may start.

    This is deliberately not another top-level intent taxonomy.  The task type
    already selected modeling or feature analysis; the LLM only decides whether
    the current utterance is an unconditional workflow command and, for
    modeling, extracts the bounded controls already owned by modeling setup.
    Questions, conditions, refusals, malformed replies, and low confidence all
    leave the task unchanged.
    """

    if runtime.llm_client is None:
        return (None, "当前专属工作流没有可用的语义理解模型，未创建计划。")

    is_modeling = task.task_type == TASK_TYPE_MODELING
    param_schema = _modeling_intake_param_schema(task) if is_modeling else []
    gate_context = (
        (
            "建模工作流首轮规格收集：任务类型已经由用户选择为 modeling。"
            "confirm 只表示无条件创建一份可审阅的建模计划；adjust 只抽取下方"
            "已声明的建模控件并创建可审阅计划，不执行计划。问题、条件、拒绝、"
            "切换工作流或不明确表达必须 clarify。"
        )
        if is_modeling
        else (
            "特征分析工作流首轮进入确认：任务类型已经由用户选择为 "
            "feature_analysis。只有用户明确、即时、无条件要求进入该工作流时"
            "才能 confirm；问题、条件、拒绝、参数变更、切换工作流或不明确表达"
            "必须 clarify。confirm 只创建可审阅计划，不执行计划。"
        )
    )
    try:
        route = route_instruction(
            runtime.llm_client,
            gate_context=gate_context,
            instruction=instruction,
            param_schema=param_schema,
            strict_contract=True,
        )
    except LLMClientError:
        return (None, "专属工作流语义理解失败，未创建计划。")

    reason = str(route.get("reason") or "").strip() or "首轮工作流授权不明确。"
    if (
        route.get("confidence") != "high"
        or bool(str(route.get("constraint") or "").strip())
    ):
        return (None, reason)

    action = route.get("action")
    params = route.get("params")
    if not isinstance(params, dict):
        return (None, reason)
    if action == "confirm":
        if route.get("explicit_authorization") is True and not params:
            return (dict(route), reason)
        return (None, reason)
    if (
        is_modeling
        and action == "adjust"
        and route.get("explicit_authorization") is False
        and params
        and set(params) <= _MODELING_INTAKE_PARAM_NAMES
    ):
        return (dict(route), reason)
    return (None, reason)

def _is_strategy_request_intent(text: str) -> bool:
    """Recognize standard strategy requests plus narrow tree-build shorthand."""

    return bool(
        utterance_targets_candidate_monthly_stability(text)
        or utterance_targets_model_score_comparison_v2(text)
        or utterance_targets_interactive_tree_frontier_group_materialization(
            text
        )
        or utterance_targets_scorecard_band_build(text)
        or utterance_targets_scorecard_cutoff_selection(text)
        or utterance_targets_strategy_dsl_delivery(text)
        or utterance_targets_strategy_pool_materialize(text)
        or utterance_targets_strategy_pool_stability(text)
        or _STRATEGY_AUTOMATIC_TREE_SHORTHAND_RE.search(text)
        or (
            _STRATEGY_REQUEST_ACTION_RE.search(text)
            and _STRATEGY_REQUEST_SUBJECT_RE.search(text)
        )
    )

def _handle_structured_strategy_request_turn(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    *,
    user_text: str | None,
    strategy_request: Mapping[str, object],
) -> dict:
    """Validate and run one LLM-free Candidate Lab request.

    The HTTP adapter constrains this envelope, while this core boundary keeps
    direct callers fail-closed. Platform-owned bindings are still selected by
    the same preparation path used by natural-language requests.
    """

    if task.task_type != TASK_TYPE_STRATEGY:
        raise DriverError("strategy_request 只能用于 strategy 类型任务。")
    request_kind = strategy_request.get("request_kind")
    workflow = strategy_request.get("workflow")
    if request_kind == "standard_workflow":
        if (
            not isinstance(workflow, str)
            or workflow not in _MANUAL_STRATEGY_WORKFLOWS
        ):
            raise DriverError(
                "strategy_request 包含未开放的 Candidate Lab workflow。"
            )
        source_metadata = {
            "intent": "strategy_request",
            "request_source": "manual_ui",
            "workflow": workflow,
        }
    elif request_kind == "strategy_lifecycle":
        operation = strategy_request.get("operation")
        strategy_type = strategy_request.get("strategy_type")
        if operation != "adopt" or not isinstance(strategy_type, str):
            raise DriverError(
                "Candidate Lab 生命周期入口只开放本地采纳复核。"
            )
        source_metadata = {
            "intent": "strategy_request",
            "request_source": "manual_ui",
            "operation": operation,
            "strategy_type": strategy_type,
        }
    else:
        raise DriverError("strategy_request.request_kind 无效。")

    conversation = repo.list_agent_messages(task.id)
    if _active_plan(runtime.plan_repo, task.id) is not None:
        raise DriverError("当前策略任务已有进行中的计划，不能启动新的 Candidate Lab 请求。")
    if latest_open_gate(conversation) is not None:
        raise DriverError("当前策略任务有待处理确认门，不能启动新的 Candidate Lab 请求。")

    pending = _latest_strategy_request_pending(conversation)
    if pending is not None:
        _invalidate_pending_strategy_request(runtime, task, pending)

    preview = None
    preview_error = None
    try:
        preview = (
            _strategy_sample_design_dataset_preview(runtime, task)
            if request_kind == "standard_workflow"
            and workflow in {"strategy_sample_design", "strategy_sample_design_v2"}
            else (
                _strategy_impact_cube_dataset_preview(runtime, task)
                if request_kind == "standard_workflow"
                and workflow == "strategy_impact_cube"
                else _strategy_dataset_preview(runtime, task)
            )
        )
    except StrategySetupError as exc:
        preview_error = str(exc)

    compilation = validate_strategy_request(
        strategy_request,
        allowed_columns=_strategy_request_allowed_columns(preview),
        target_col=None if preview is None else preview.target_col,
    )
    if compilation.draft is None:
        repo.add_agent_message(
            task.id,
            role="user",
            stage="chat",
            content=str(user_text or "").strip(),
            metadata=source_metadata,
        )
        return _strategy_request_clarification_response(
            repo,
            task,
            code=(
                compilation.clarification_code
                or "strategy_request_needs_clarification"
            ),
            message=compilation.clarification or "请修正 Candidate Lab 策略请求。",
            fields=compilation.clarification_fields,
        )
    draft = compilation.draft
    if request_kind == "standard_workflow":
        if (
            not isinstance(draft, StandardWorkflowRequestDraft)
            or draft.workflow != workflow
        ):
            raise DriverError("Candidate Lab 请求未编译为预期的标准 Workflow。")
    elif (
        not isinstance(draft, StrategyRequestDraft)
        or draft.operation != "adopt"
        or draft.strategy_type != strategy_request.get("strategy_type")
    ):
        raise DriverError("Candidate Lab 请求未编译为预期的本地采纳复核。")

    if isinstance(draft, StandardWorkflowRequestDraft):
        draft_inputs = draft.to_dict()["workflow_inputs"]
        if draft.workflow == "strategy_project_context":
            source_metadata["structured_request_sha256"] = (
                strategy_project_context_structured_request_sha256(
                    as_of=draft_inputs["as_of"],
                    scope=draft_inputs.get("scope"),
                    business_context=draft_inputs["business_context"],
                    explicit_unavailable=draft_inputs["explicit_unavailable"],
                    external_report_filenames=draft_inputs[
                        "external_report_filenames"
                    ],
                )
            )
    source_message = repo.add_agent_message(
        task.id,
        role="user",
        stage="chat",
        content=str(user_text or "").strip(),
        metadata=source_metadata,
    )

    if (
        isinstance(draft, StandardWorkflowRequestDraft)
        and draft.workflow
        in {"strategy_sample_design", "strategy_sample_design_v2"}
    ):
        try:
            _confirm_manual_sample_design_time_semantics(
                runtime,
                task,
                draft=draft,
                preview=preview,
            )
            preview = _strategy_sample_design_dataset_preview(runtime, task)
        except StrategySetupError as exc:
            return _strategy_request_clarification_response(
                repo,
                task,
                code="strategy_sample_design_time_semantics_invalid",
                message=str(exc),
                fields=("field_bindings.time_field",),
            )

    preflight = _strategy_request_preflight(runtime, task, draft)
    if preflight is not None:
        code, message = preflight
        return _strategy_request_clarification_response(
            repo,
            task,
            code=code,
            message=message,
        )
    if _strategy_request_requires_dataset(draft) and preview is None:
        return _strategy_request_clarification_response(
            repo,
            task,
            code="strategy_dataset_context_required",
            message=preview_error or "当前策略操作需要一个任务内样本。",
        )
    if _strategy_request_requires_target(draft) and (
        preview is None or not preview.target_col
    ):
        return _strategy_request_clarification_response(
            repo,
            task,
            code="strategy_target_context_required",
            message="当前策略操作需要明确的二元目标列，请先在任务中指定 target_col。",
        )

    return _prepare_and_run_validated_strategy_request(
        runtime,
        repo,
        task,
        draft,
        preview=preview,
        auto_start=True,
        source_message=source_message,
    )

def _maybe_handle_strategy_request_turn(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    *,
    user_text: str | None,
    force_intent: bool = False,
) -> dict | None:
    """Compile a natural-language strategy request and route it safely.

    A strict compiler result plus preflight authorizes reversible execution. The
    plan overview is auto-resumed in the same request and platform AUTO safety
    still stops at real human-responsibility gates (adoption/disposition). Legacy
    persisted request confirmations remain readable for compatibility only.
    """

    if task.task_type != TASK_TYPE_STRATEGY or runtime.llm_client is None:
        return None
    text = str(user_text or "").strip()
    if not text:
        return None

    conversation = repo.list_agent_messages(task.id)
    if latest_unresolved_workflow_failure(
        conversation,
        workflow=task.task_type,
    ) is not None and is_explicit_workflow_retry(text):
        # The recovery branch deliberately returns None for an explicit retry;
        # do not reinterpret that command as a brand-new strategy request.
        return None
    pending = _latest_strategy_request_pending(conversation)
    if pending is not None and is_confirm(text):
        repo.add_agent_message(
            task.id,
            role="user",
            stage="chat",
            content=text,
            metadata={"intent": "strategy_request_confirmation"},
        )
        if (
            _active_plan(runtime.plan_repo, task.id) is not None
            or latest_open_gate(conversation) is not None
        ):
            _invalidate_pending_strategy_request(runtime, task, pending)
            return _strategy_request_clarification_response(
                repo,
                task,
                code="strategy_request_stale_confirmation",
                message="任务状态已变化，旧策略草案已失效；请完成当前计划后重新发起。",
            )
        return _run_confirmed_strategy_request(runtime, repo, task, pending)
    if pending is not None and _STRATEGY_REQUEST_CANCEL_RE.search(text):
        repo.add_agent_message(
            task.id,
            role="user",
            stage="chat",
            content=text,
            metadata={"intent": "strategy_request_cancel"},
        )
        try:
            PendingStrategyRequestRepository(runtime.settings.db_path).cancel(
                task_id=task.id,
                request_id=str(pending.get("request_id") or ""),
                expected_payload_sha256=str(pending.get("payload_sha256") or ""),
            )
        except (
            PendingStrategyRequestConflictError,
            PendingStrategyRequestNotFoundError,
            ValueError,
        ):
            return _strategy_request_clarification_response(
                repo,
                task,
                code="strategy_request_stale_cancellation",
                message="上一份策略草案已被处理或失效，请重新发起需要执行的操作。",
            )
        repo.add_agent_message(
            task.id,
            role="assistant",
            stage="chat",
            content="已取消上一份策略执行草案，没有创建计划或执行工具。",
            metadata={"intent": "strategy_request_cancelled"},
        )
        return join_turn_response(repo, task.id)
    if pending is not None:
        # Any non-confirm/non-cancel reply replaces the confirmation context.
        # Make the opaque row terminal as well as hiding the old message ref so
        # abandoned drafts cannot accumulate as apparently actionable state.
        _invalidate_pending_strategy_request(runtime, task, pending)

    nan_confirmation = _latest_strategy_nan_label_confirmation(conversation)
    if nan_confirmation is not None:
        if _STRATEGY_DROP_NAN_CANCEL_RE.search(text):
            repo.add_agent_message(
                task.id,
                role="user",
                stage="chat",
                content=text,
                metadata={"intent": "strategy_drop_nan_labels_cancel"},
            )
            repo.add_agent_message(
                task.id,
                role="assistant",
                stage="chat",
                content="已取消本次策略执行；未应用任何空标签排除口径，也没有创建计划。",
                metadata={
                    "intent": "strategy_drop_nan_labels_cancelled",
                    "kind": "clarification",
                    "code": "strategy_drop_nan_labels_cancelled",
                },
            )
            return join_turn_response(repo, task.id)
        if _STRATEGY_DROP_NAN_CONFIRM_RE.search(text):
            repo.add_agent_message(
                task.id,
                role="user",
                stage="chat",
                content=text,
                metadata={"intent": "strategy_drop_nan_labels_confirm"},
            )
            return _resume_strategy_after_nan_label_confirmation(
                runtime,
                repo,
                task,
                nan_confirmation,
            )
        if is_confirm(text):
            repo.add_agent_message(
                task.id,
                role="user",
                stage="chat",
                content=text,
                metadata={"intent": "strategy_drop_nan_labels_ambiguous"},
            )
            return _repeat_strategy_nan_label_clarification(
                repo,
                task,
                nan_confirmation,
            )

    # An explicit project-context command is a new workflow request, even when
    # it also happens to answer one pending field.  Only bare follow-up answers
    # use the shortcut below; otherwise a refresh could be mistaken for an
    # answer and silently skip the newly supplied as-of/scope controls.
    project_context_answer = (
        None
        if (
            force_intent
            or
            utterance_targets_strategy_project_context(text)
            or _is_strategy_request_intent(text)
        )
        else _maybe_handle_project_context_missing_answer(
            runtime,
            repo,
            task,
            text=text,
            conversation=conversation,
        )
    )
    if project_context_answer is not None:
        return project_context_answer

    if not force_intent and not _is_strategy_request_intent(text):
        return None
    if _active_plan(runtime.plan_repo, task.id) is not None:
        return None
    if latest_open_gate(conversation) is not None:
        return None

    if _STRATEGY_REQUEST_NON_EXECUTION_RE.search(
        text
    ) and not _STRATEGY_POOL_COMPILE_REQUEST_RE.search(text):
        repo.add_agent_message(
            task.id,
            role="user",
            stage="chat",
            content=text,
            metadata={"intent": "strategy_preview_only"},
        )
        repo.add_agent_message(
            task.id,
            role="assistant",
            stage="chat",
            content=(
                "已按仅预览/讨论处理：本轮不会编译执行草案、创建计划或调用策略工具。"
                "需要实际执行时，请另发一条明确的执行请求。"
            ),
            metadata={
                "intent": "strategy_preview_only",
                "kind": "clarification",
                "code": "strategy_execution_not_authorized",
            },
        )
        return {
            "task_id": task.id,
            "status": "preview_only",
            "code": "strategy_execution_not_authorized",
            "messages": repo.list_agent_messages(task.id),
        }

    source_message = repo.add_agent_message(
        task.id,
        role="user",
        stage="chat",
        content=text,
        metadata={
            "intent": "strategy_request",
            "request_source": "agent_nl",
        },
    )
    preview = None
    preview_error = None
    is_project_context_request = utterance_targets_strategy_project_context(text)
    is_sample_design_request = utterance_targets_strategy_sample_design(text)
    is_report_bundle_v2_request = utterance_targets_strategy_report_bundle_v2(
        text
    )
    is_candidate_stability_request = (
        utterance_targets_candidate_monthly_stability(text)
    )
    is_pool_validation_request = (
        _STRATEGY_POOL_VALIDATION_REQUEST_RE.search(text) is not None
    )
    is_scorecard_request = (
        utterance_targets_scorecard_band_build(text)
        or utterance_targets_scorecard_cutoff_selection(text)
    )
    is_impact_cube_request = utterance_targets_strategy_impact_cube(text)
    is_pool_stability_request = utterance_targets_strategy_pool_stability(text)
    is_model_evidence_v2_request = (
        _STRATEGY_MODEL_EVIDENCE_V2_REQUEST_RE.search(text) is not None
    )
    try:
        preview = (
            None
            if (
                is_project_context_request
                or is_model_evidence_v2_request
                or is_report_bundle_v2_request
                or is_candidate_stability_request
                or is_pool_validation_request
                or is_pool_stability_request
                or is_scorecard_request
            )
            else (
                _strategy_impact_cube_dataset_preview(runtime, task)
                if is_impact_cube_request
                else (
                    _strategy_pool_impact_dataset_preview(runtime, task)
                    if _STRATEGY_POOL_IMPACT_REQUEST_RE.search(text)
                    else (
                        _strategy_sample_design_dataset_preview(runtime, task)
                        if is_sample_design_request
                        else _strategy_dataset_preview(runtime, task)
                    )
                )
            )
        )
    except StrategySetupError as exc:
        preview_error = str(exc)

    if is_sample_design_request and preview is None:
        return _strategy_request_clarification_response(
            repo,
            task,
            code=(
                "strategy_sample_design_target_invalid"
                if preview_error and "必须是数值 0/1 或真实空值" in preview_error
                else "strategy_sample_design_workspace_required"
            ),
            message=preview_error or "样本设计要求先确认活动 DataWorkspace。",
        )

    compilation = compile_strategy_request(
        text,
        allowed_columns=_strategy_request_allowed_columns(preview),
        target_col=None if preview is None else preview.target_col,
        llm=runtime.llm_client,
    )
    if compilation.draft is None:
        return _strategy_request_clarification_response(
            repo,
            task,
            code=(
                compilation.clarification_code or "strategy_request_needs_clarification"
            ),
            message=compilation.clarification or "请补充策略操作、策略类型和业务口径。",
            fields=compilation.clarification_fields,
        )
    draft = compilation.draft
    if (
        isinstance(draft, StandardWorkflowRequestDraft)
        and draft.workflow
        in {"strategy_sample_design", "strategy_sample_design_v2"}
    ):
        # Natural-language and Candidate Lab requests share the same validated
        # draft contract.  Once the exact time_field has passed grounding and
        # column validation, persist its date role before preflight so the
        # deterministic materializer sees the same confirmed semantics in both
        # entry paths.
        try:
            _confirm_manual_sample_design_time_semantics(
                runtime,
                task,
                draft=draft,
                preview=preview,
            )
            preview = _strategy_sample_design_dataset_preview(runtime, task)
        except StrategySetupError as exc:
            return _strategy_request_clarification_response(
                repo,
                task,
                code="strategy_sample_design_time_semantics_invalid",
                message=str(exc),
                fields=("field_bindings.time_field",),
            )
    preflight = _strategy_request_preflight(runtime, task, draft)
    if preflight is not None:
        code, message = preflight
        return _strategy_request_clarification_response(
            repo,
            task,
            code=code,
            message=message,
        )
    requires_dataset = _strategy_request_requires_dataset(draft)
    if requires_dataset and preview is None:
        return _strategy_request_clarification_response(
            repo,
            task,
            code="strategy_dataset_context_required",
            message=preview_error or "当前策略操作需要一个任务内样本。",
        )
    if _strategy_request_requires_target(draft) and (
        preview is None or not preview.target_col
    ):
        return _strategy_request_clarification_response(
            repo,
            task,
            code="strategy_target_context_required",
            message="当前策略操作需要明确的二元目标列，请先在任务中指定 target_col。",
        )

    return _prepare_and_run_validated_strategy_request(
        runtime,
        repo,
        task,
        draft,
        preview=preview,
        auto_start=True,
        source_message=source_message,
    )

def _maybe_handle_project_context_missing_answer(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    *,
    text: str,
    conversation: Sequence[Mapping],
) -> dict | None:
    """Turn a direct answer to a pending context question into an audited refresh.

    The answer is retained as user evidence.  It is never parsed into a
    deterministic metric here; the Tool labels such values as user-provided
    and unverified, while an explicit unavailable answer remains null.
    """

    if (
        _active_plan(runtime.plan_repo, task.id) is not None
        or latest_open_gate(list(conversation)) is not None
    ):
        return None
    try:
        current = StrategyProjectContextRepository(
            runtime.settings.db_path
        ).get_current(task.id)
    except (StrategyProjectContextDataError, KeyError, TypeError, ValueError):
        return None
    if current is None:
        return None
    pending = [
        record
        for record in current["state"]["missing_information_records"]
        if record["status"] == "pending"
    ]
    if not pending:
        return None
    pending_paths = {record["field_path"] for record in pending}
    mentioned = [
        field_path
        for field_path, pattern in _PROJECT_CONTEXT_ANSWER_PATTERNS.items()
        if field_path in pending_paths and pattern.search(text)
    ]
    unavailable = _PROJECT_CONTEXT_UNAVAILABLE_ANSWER_RE.search(text) is not None
    if not mentioned and unavailable:
        if _PROJECT_CONTEXT_ALL_PENDING_RE.search(text):
            mentioned = sorted(pending_paths)
        elif len(pending_paths) == 1:
            mentioned = list(pending_paths)
        else:
            repo.add_agent_message(
                task.id,
                role="user",
                stage="chat",
                content=text,
                metadata={"intent": "strategy_project_context_answer_ambiguous"},
            )
            return _strategy_request_clarification_response(
                repo,
                task,
                code="strategy_project_context_answer_field_required",
                message=(
                    "请说明哪些字段暂时没有；可点名通过率、风险率、业务量、"
                    "收益成本、样本成熟度或历史策略，也可以明确说“以上全部暂时没有”。"
                ),
                fields=tuple(sorted(pending_paths)),
            )
    if not mentioned:
        return None

    source_message = repo.add_agent_message(
        task.id,
        role="user",
        stage="chat",
        content=text,
        metadata={
            "intent": "strategy_project_context_answer",
            "field_paths": list(mentioned),
            "answer_status": "unavailable" if unavailable else "provided",
        },
    )
    workflow_inputs = {
        "as_of": current["state"]["as_of"],
        "business_context": (
            {} if unavailable else {field_path: text for field_path in mentioned}
        ),
        "explicit_unavailable": list(mentioned) if unavailable else [],
        "external_report_filenames": [],
    }
    draft = StandardWorkflowRequestDraft(
        workflow="strategy_project_context",
        workflow_inputs=workflow_inputs,
    )
    return _prepare_and_run_validated_strategy_request(
        runtime,
        repo,
        task,
        draft,
        preview=None,
        auto_start=True,
        source_message=source_message,
    )

def _prepare_and_run_validated_strategy_request(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    draft: CompiledStrategyRequestDraft,
    *,
    preview,
    auto_start: bool,
    drop_nan_labels: bool = False,
    expected_pool_binding: Mapping | None = None,
    source_message: Mapping | None = None,
) -> dict:
    """Bind current evidence, resolve the NaN policy, then instantiate once."""

    expected_impact_cube_sample_binding = None
    if (
        isinstance(draft, StandardWorkflowRequestDraft)
        and draft.workflow == "strategy_impact_cube"
        and preview is not None
    ):
        identity = getattr(preview, "identity", None)
        if not isinstance(identity, Mapping):
            return _strategy_request_clarification_response(
                repo,
                task,
                code="strategy_impact_cube_sample_invalid",
                message=(
                    "ImpactCube 编译预览缺少认证 SampleDesign 身份；"
                    "请重新固化样本设计后重试。"
                ),
            )
        expected_impact_cube_sample_binding = dict(identity)

    if (
        isinstance(draft, StandardWorkflowRequestDraft)
        and draft.workflow_inputs.get("drop_nan_labels") is True
    ):
        # This boolean has already passed exact utterance grounding in the
        # compiler; it is the user's explicit authorization, not an LLM default.
        drop_nan_labels = True
    requires_dataset = _strategy_request_requires_dataset(draft)
    is_pool_impact = (
        isinstance(draft, StandardWorkflowRequestDraft)
        and draft.workflow in _STRATEGY_POOL_MEASUREMENT_WORKFLOWS
    )
    is_sample_design = (
        isinstance(draft, StandardWorkflowRequestDraft)
        and draft.workflow
        in {"strategy_sample_design", "strategy_sample_design_v2"}
    )
    context = None
    if requires_dataset:
        try:
            context = (
                _strategy_pool_impact_dataset_context(runtime, task)
                if is_pool_impact
                else (
                    _strategy_sample_design_dataset_context(runtime, task)
                    if is_sample_design
                    else _strategy_dataset_context(
                        runtime,
                        task,
                        require_target=_strategy_request_requires_target(draft),
                    )
                )
            )
        except StrategySetupError as exc:
            return append_join_error(repo, task.id, str(exc))
        if _is_automatic_tree_build_draft(draft):
            if preview is None or not _strategy_dataset_binding_matches(
                runtime,
                task,
                preview=preview,
                context=context,
            ):
                return _strategy_request_clarification_response(
                    repo,
                    task,
                    code="strategy_dataset_context_changed",
                    message=(
                        "策略样本在编译与活动工作区绑定之间发生变化；本次请求未执行，"
                        "请基于当前数据重新描述。"
                    ),
                )
            try:
                preview, context = _ensure_automatic_tree_active_workspace(
                    runtime,
                    task,
                    preview=preview,
                    context=context,
                )
            except StrategySetupError as exc:
                return _strategy_request_clarification_response(
                    repo,
                    task,
                    code="automatic_tree_active_workspace_required",
                    message=str(exc),
                )
        if preview is None or not _strategy_dataset_binding_matches(
            runtime,
            task,
            preview=preview,
            context=context,
            use_confirmed_workspace_target=is_pool_impact,
            use_sample_design_workspace=is_sample_design,
        ):
            return _strategy_request_clarification_response(
                repo,
                task,
                code="strategy_dataset_context_changed",
                message=(
                    "策略样本在编译与计划创建之间发生变化；本次请求未执行，"
                    "请基于当前数据重新描述。"
                ),
            )

    if (
        isinstance(draft, StrategyRequestDraft)
        and draft.operation == "adopt"
    ):
        try:
            drop_nan_labels = _inherit_strategy_sample_drop_nan_policy(
                runtime,
                task,
                context=context,
            )
        except _StrategyV2EvidenceSetupError as exc:
            return _strategy_request_clarification_response(
                repo,
                task,
                code=exc.code,
                message=str(exc),
            )
        except StrategySetupError as exc:
            return append_join_error(repo, task.id, str(exc))

    if _strategy_request_requires_complete_labels(draft):
        assert context is not None and context.target_col
        try:
            n_total, n_nan = _strategy_target_nan_stats(runtime, context)
        except StrategySetupError as exc:
            return _strategy_request_clarification_response(
                repo,
                task,
                code="strategy_target_labels_invalid",
                message=str(exc),
                fields=("target_col",),
            )
        if n_nan and not drop_nan_labels:
            return _strategy_nan_label_clarification_response(
                runtime,
                repo,
                task,
                draft=draft,
                context=context,
                n_total=n_total,
                n_nan=n_nan,
            )
        inherits_sample_drop_nan_policy = (
            isinstance(draft, StandardWorkflowRequestDraft)
            and draft.workflow in _STRATEGY_SAMPLE_BOUND_CANDIDATE_WORKFLOWS
            and "drop_nan_labels" not in draft.workflow_inputs
        )
        if inherits_sample_drop_nan_policy:
            try:
                drop_nan_labels = (
                    _inherit_strategy_sample_drop_nan_policy(
                        runtime,
                        task,
                        context=context,
                    )
                )
            except _StrategyV2EvidenceSetupError as exc:
                return _strategy_request_clarification_response(
                    repo,
                    task,
                    code=exc.code,
                    message=str(exc),
                )
            except StrategySetupError as exc:
                return append_join_error(repo, task.id, str(exc))

    try:
        return _run_validated_strategy_request(
            runtime,
            repo,
            task,
            draft,
            context=context,
            auto_start=auto_start,
            drop_nan_labels=drop_nan_labels,
            expected_pool_binding=expected_pool_binding,
            expected_impact_cube_sample_binding=(
                expected_impact_cube_sample_binding
            ),
            source_message=source_message,
        )
    except _StrategyV2EvidenceSetupError as exc:
        return _strategy_request_clarification_response(
            repo,
            task,
            code=exc.code,
            message=str(exc),
        )
    except _StrategySampleDesignRequiredError as exc:
        return _strategy_request_clarification_response(
            repo,
            task,
            code="strategy_sample_design_required",
            message=str(exc),
        )
    except StrategyWorkflowValidationError as exc:
        return _strategy_request_clarification_response(
            repo,
            task,
            code=exc.code,
            message=str(exc),
        )
    except StrategySetupError as exc:
        return append_join_error(repo, task.id, str(exc))
    except DriverError:
        raise
    except Exception as exc:
        return append_join_error(repo, task.id, f"策略请求执行出错：{exc}")

def _run_validated_strategy_request(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    draft: CompiledStrategyRequestDraft,
    *,
    context,
    auto_start: bool,
    drop_nan_labels: bool,
    expected_pool_binding: Mapping | None = None,
    expected_impact_cube_sample_binding: Mapping | None = None,
    source_message: Mapping | None = None,
) -> dict:
    """Route one already-validated draft without another execution confirmation."""

    if isinstance(draft, StandardWorkflowRequestDraft):
        workflow_inputs = draft.to_dict()["workflow_inputs"]

        def bind_sample_design(allow_native_risk_development: bool):
            if context is None:
                raise StrategyWorkflowValidationError(
                    "当前 Workflow 需要任务内数据上下文。",
                    code="strategy_dataset_required",
                )
            return _latest_matching_strategy_sample_design_ref(
                runtime,
                task,
                context=context,
                drop_nan_labels=bool(drop_nan_labels),
                allow_native_risk_development=allow_native_risk_development,
            )

        def validate_strategy_ref(
            strategy_id: str,
            allowed_types: frozenset[str],
        ) -> None:
            meta = StrategyRepository(runtime.settings.db_path).get_strategy_meta(
                strategy_id
            )
            if meta is None or meta.get("task_id") != task.id:
                raise StrategyWorkflowValidationError(
                    "没有在当前任务中找到要关联的策略，不能跨任务挂载矩阵产物。",
                    code="strategy_not_owned_by_task",
                    fields=("strategy_id",),
                )
            if meta.get("strategy_type") not in allowed_types:
                raise StrategyWorkflowValidationError(
                    "额度定价矩阵只能关联当前任务中的额度或定价策略。",
                    code="strategy_type_mismatch",
                    fields=("strategy_id",),
                )

        def bind_model_score_comparison(
            population: str,
            partition: str,
        ) -> Mapping[str, object]:
            return _model_score_comparison_plan_slots(
                runtime,
                task,
                population=population,
                partition=partition,
            )

        def bind_workflow_evidence(
            workflow_id: str,
            normalized_inputs: Mapping[str, object],
        ) -> Mapping[str, object]:
            def evidence_only(
                bound_slots: Mapping[str, object],
                *canonical_fields: str,
                canonical_overrides: Mapping[str, object] | None = None,
            ) -> dict[str, object]:
                expected_user_slots = {
                    field: workflow_inputs[field]
                    for field in canonical_fields
                    if field in workflow_inputs
                }
                if canonical_overrides is not None:
                    expected_user_slots.update(canonical_overrides)
                return _platform_evidence_only(
                    workflow_id,
                    bound_slots,
                    expected_user_slots=expected_user_slots,
                )

            if workflow_id == "strategy_project_context":
                bound = _strategy_project_context_plan_slots(
                    runtime,
                    task,
                    draft,
                    source_message=source_message,
                )
                return evidence_only(
                    bound,
                    "as_of",
                    "scope",
                    "business_context",
                    "explicit_unavailable",
                    "external_report_filenames",
                    canonical_overrides={
                        "scope": workflow_inputs.get("scope")
                    },
                )
            if workflow_id == "strategy_sample_design":
                if context is None:
                    raise StrategySetupError(
                        "策略样本设计需要确认的活动 DataWorkspace 和二元目标列。"
                    )
                bound = _strategy_sample_design_plan_slots(
                    runtime,
                    task,
                    draft,
                    context=context,
                    drop_nan_labels=drop_nan_labels,
                )
                return evidence_only(
                    bound,
                    *normalized_inputs,
                    canonical_overrides={
                        "drop_nan_labels": bool(drop_nan_labels)
                    },
                )
            if workflow_id == "strategy_sample_design_v2":
                if context is None:
                    raise StrategySetupError(
                        "V2 策略样本设计需要确认的活动 DataWorkspace 和二元目标列。"
                    )
                bound = _strategy_sample_design_v2_plan_slots(
                    runtime,
                    task,
                    draft,
                    context=context,
                    drop_nan_labels=drop_nan_labels,
                )
                return evidence_only(
                    bound,
                    *normalized_inputs,
                    canonical_overrides={
                        "drop_nan_labels": bool(drop_nan_labels)
                    },
                )
            if workflow_id == "strategy_model_evidence_v2":
                return _strategy_model_evidence_v2_plan_slots(
                    runtime,
                    task,
                    verify_current=True,
                )
            if workflow_id == "strategy_dsl_delivery":
                if context is None:
                    raise StrategySetupError(
                        "策略代码交付需要当前任务内唯一且已认证的数据集。"
                    )
                return _strategy_dsl_delivery_plan_slots(
                    runtime,
                    task,
                    draft,
                    context=context,
                )
            if workflow_id == "strategy_report_bundle_v2":
                bound = _strategy_report_bundle_v2_plan_slots(
                    runtime,
                    task,
                    draft,
                    source_message=source_message,
                )
                return evidence_only(bound, "title", "status")
            if workflow_id == "univariate_candidate_analysis":
                return _bind_univariate_dataset_evidence(
                    runtime,
                    task,
                    normalized_inputs,
                    context=context,
                    drop_nan_labels=bool(drop_nan_labels),
                )
            if workflow_id == "univariate_candidate_refinement":
                source_candidate_id = normalized_inputs.get(
                    "source_candidate_id"
                )
                if source_candidate_id is not None:
                    try:
                        return _bind_candidate_source_artifact_evidence(
                            runtime,
                            task_id=task.id,
                            candidate_id=str(source_candidate_id),
                            workflow_inputs=normalized_inputs,
                        )
                    except StrategySetupError as exc:
                        raise StrategyWorkflowValidationError(
                            str(exc),
                            code="strategy_candidate_source_required",
                        ) from exc
                return _bind_univariate_dataset_evidence(
                    runtime,
                    task,
                    normalized_inputs,
                    context=context,
                    drop_nan_labels=bool(drop_nan_labels),
                )
            if workflow_id == "candidate_monthly_stability":
                try:
                    return _bind_candidate_monthly_stability_evidence(
                        runtime,
                        task,
                        normalized_inputs,
                    )
                except StrategySetupError as exc:
                    message = str(exc)
                    code = (
                        "candidate_monthly_stability_month_required"
                        if "月份字段" in message or "month field" in message
                        else "candidate_monthly_stability_binding_required"
                    )
                    raise StrategyWorkflowValidationError(
                        message,
                        code=code,
                    ) from exc
            if workflow_id == "scorecard_model_score_evidence_build":
                return _bind_scorecard_model_score_evidence(
                    runtime,
                    task,
                )
            if workflow_id == "scorecard_band_build":
                return _bind_scorecard_band_evidence(runtime, task)
            if workflow_id == "scorecard_cutoff_selection":
                return _bind_scorecard_cutoff_evidence(
                    runtime,
                    task_id=task.id,
                    workflow_inputs=normalized_inputs,
                )
            if workflow_id in {
                "automatic_tree_candidate_build",
                "cross_matrix_analysis",
            }:
                return _bind_univariate_dataset_evidence(
                    runtime,
                    task,
                    normalized_inputs,
                    context=context,
                    drop_nan_labels=bool(drop_nan_labels),
                )
            if workflow_id == "automatic_tree_apply":
                if context is None:
                    raise StrategySetupError(
                        "自动树全量写回需要当前活动 DataWorkspace。"
                    )
                bound = _automatic_tree_apply_slots(
                    runtime,
                    task_id=task.id,
                    draft=draft,
                    context=context,
                )
                return evidence_only(
                    bound,
                    "leaf_id_column",
                    "rule_id_column",
                )
            if workflow_id == "automatic_tree_leaf_materialization":
                bound = _automatic_tree_leaf_materialization_slots(
                    runtime,
                    task_id=task.id,
                    draft=draft,
                )
                return evidence_only(bound, "leaf_id", "selection_reason")
            if workflow_id == "interactive_tree_split_search":
                _interactive_tree_split_search_plan_slots(
                    runtime,
                    task_id=task.id,
                    draft=draft,
                )
                return {}
            if workflow_id == "interactive_tree_auto_continuation":
                _interactive_tree_auto_continuation_plan_slots(
                    runtime,
                    task_id=task.id,
                    draft=draft,
                )
                return {}
            if workflow_id == "interactive_tree_revision":
                if normalized_inputs.get("operation") in {
                    "adjust_split_threshold",
                    "replace_split_feature",
                }:
                    _interactive_tree_revision_plan_slots(
                        runtime,
                        task_id=task.id,
                        draft=draft,
                    )
                return {}
            if workflow_id == "voting_candidate_search":
                bound = _strategy_voting_candidate_search_plan_slots(
                    runtime,
                    task,
                    draft,
                )
                return evidence_only(bound, *normalized_inputs)
            if workflow_id == "voting_candidate_build_from_search":
                _strategy_voting_candidate_build_from_search_plan_slots(
                    runtime,
                    task,
                    draft,
                )
                return {}
            if workflow_id == "voting_candidate_build":
                bound = _strategy_voting_candidate_plan_slots(
                    runtime,
                    task,
                    draft,
                )
                return evidence_only(bound, "strategy_type", "n")
            if workflow_id == "cross_matrix_candidate_search":
                bound = _strategy_cross_candidate_search_plan_slots(
                    runtime,
                    task,
                    draft,
                )
                return evidence_only(bound, *normalized_inputs)
            if workflow_id == "cross_matrix_candidate_build_from_search":
                _strategy_cross_candidate_build_from_search_plan_slots(
                    runtime,
                    task,
                    draft,
                )
                return {}
            if workflow_id == "cross_rule_search":
                bound = _strategy_cross_rule_search_plan_slots(
                    runtime,
                    task,
                    draft,
                )
                return evidence_only(bound, *normalized_inputs)
            if workflow_id == "cross_rule_candidate_build_from_search":
                _strategy_cross_rule_candidate_build_plan_slots(
                    runtime,
                    task,
                    draft,
                )
                return {}
            if workflow_id == "cross_matrix_cell_selection":
                bound = _cross_matrix_cell_selection_slots(
                    runtime,
                    task_id=task.id,
                    draft=draft,
                )
                return evidence_only(bound, "selection_reason")
            if workflow_id in {
                "strategy_pool_add_candidate",
                "strategy_pool_remove_entry",
                "strategy_pool_set_action",
                "strategy_pool_reorder",
                "strategy_pool_compile",
            }:
                bound = _strategy_pool_plan_slots(runtime, task, draft)
                canonical_fields = {
                    "strategy_pool_add_candidate": (
                        "strategy_type",
                        "default_action",
                        "action",
                        "reason",
                    ),
                    "strategy_pool_remove_entry": ("strategy_type", "reason"),
                    "strategy_pool_set_action": (
                        "strategy_type",
                        "action",
                        "reason",
                    ),
                    "strategy_pool_reorder": ("strategy_type", "reason"),
                    "strategy_pool_compile": ("strategy_type",),
                }[workflow_id]
                return evidence_only(bound, *canonical_fields)
            if workflow_id == "strategy_pool_materialize":
                bound = _strategy_pool_materialize_plan_slots(
                    runtime,
                    task,
                    draft,
                )
                return evidence_only(bound, "strategy_type")
            if workflow_id == "strategy_pool_apply":
                bound = _strategy_pool_apply_plan_slots(runtime, task, draft)
                return evidence_only(bound, *normalized_inputs)
            if workflow_id == "strategy_pool_validation":
                bound = _strategy_pool_validation_plan_slots(
                    runtime,
                    task,
                    draft,
                )
                return evidence_only(bound, *normalized_inputs)
            if workflow_id == "strategy_pool_impact":
                if context is None:
                    raise StrategySetupError(
                        "Strategy Pool 影响测算需要活动 DataWorkspace 和确认的目标列。"
                    )
                bound = _strategy_pool_impact_plan_slots(
                    runtime,
                    task,
                    draft,
                    context=context,
                    drop_nan_labels=drop_nan_labels,
                    expected_pool_binding=expected_pool_binding,
                )
                return evidence_only(
                    bound,
                    "strategy_type",
                    "comparison_mode",
                    "baseline_strategy_id",
                )
            if workflow_id == "strategy_impact_cube":
                bound = _strategy_impact_cube_plan_slots(
                    runtime,
                    task,
                    draft,
                    expected_sample_binding=(
                        expected_impact_cube_sample_binding
                    ),
                )
                return evidence_only(bound, "strategy_type")
            if workflow_id == "strategy_pool_stability":
                bound = _strategy_pool_stability_plan_slots(
                    runtime,
                    task,
                    draft,
                )
                return evidence_only(bound, "strategy_type")
            raise StrategyWorkflowValidationError(
                f"{workflow_id} 没有声明平台证据绑定适配器。",
                code="strategy_workflow_evidence_binding_unsupported",
            )

        prepared = prepare_strategy_plan(
            draft.workflow,
            workflow_inputs,
            context=StrategyWorkflowPreparationContext(
                dataset_id=None if context is None else context.dataset_id,
                drop_nan_labels=bool(drop_nan_labels),
                bind_sample_design=(None if context is None else bind_sample_design),
                validate_strategy_ref=validate_strategy_ref,
                bind_model_score_comparison=bind_model_score_comparison,
                bind_workflow_evidence=bind_workflow_evidence,
            ),
        )
        start_kwargs = {}
        if prepared.success_criteria:
            start_kwargs["success_criteria"] = [
                dict(criterion) for criterion in prepared.success_criteria
            ]
        return _start_confirmed_strategy_plan(
            runtime,
            repo,
            task,
            template_id=prepared.template_id,
            slots=prepared.to_runtime_slots(),
            auto_start=auto_start,
            **start_kwargs,
        )

    if (
        isinstance(draft, StrategyRequestDraft)
        and draft.strategy_spec is None
        and draft.operation == "report"
    ):
        return _start_confirmed_strategy_plan(
            runtime,
            repo,
            task,
            template_id="stored_strategy_report",
            slots={"strategy_id": draft.strategy_id},
            auto_start=auto_start,
        )

    if context is None:
        raise StrategySetupError("当前策略操作需要任务内数据上下文。")

    if _is_auto_candidate_draft(draft):
        slots = _candidate_strategy_slots(context, draft)
        slots["sample_design_ref"] = _latest_matching_strategy_sample_design_ref(
            runtime,
            task,
            context=context,
            drop_nan_labels=bool(drop_nan_labels),
            allow_native_risk_development=True,
        )
        return _start_confirmed_strategy_plan(
            runtime,
            repo,
            task,
            template_id="deterministic_strategy_candidate_development",
            slots=_strategy_slots_with_drop_nan(
                slots,
                drop_nan_labels,
            ),
            auto_start=auto_start,
        )

    if draft.strategy_spec is not None:
        slots = (
            {
                "dataset_id": context.dataset_id,
                "strategy_spec": draft.to_dict()["strategy_spec"],
            }
            if draft.operation == "apply"
            else _typed_strategy_slots(context, draft)
        )
        template_id = {
            "develop": "typed_strategy_build",
            "apply": "typed_strategy_apply",
            "analyze": "typed_strategy_evaluation",
            "backtest": "typed_strategy_evaluation",
        }[draft.operation]
        if draft.operation in {"analyze", "backtest"}:
            slots["sample_design_ref"] = (
                _latest_matching_strategy_sample_design_ref(
                    runtime,
                    task,
                    context=context,
                    drop_nan_labels=bool(drop_nan_labels),
                    allow_native_risk_development=True,
                )
            )
        return _start_confirmed_strategy_plan(
            runtime,
            repo,
            task,
            template_id=template_id,
            slots=_strategy_slots_with_drop_nan(slots, drop_nan_labels),
            success_criteria=_strategy_request_success_criteria(draft),
            auto_start=auto_start,
        )

    if draft.operation in _STORED_EVALUATION_OPERATIONS:
        slots = _stored_strategy_slots(context, draft)
        slots["sample_design_ref"] = _latest_matching_strategy_sample_design_ref(
            runtime,
            task,
            context=context,
            drop_nan_labels=bool(drop_nan_labels),
            allow_native_risk_development=True,
        )
        return _start_confirmed_strategy_plan(
            runtime,
            repo,
            task,
            template_id="stored_strategy_evaluation",
            slots=_strategy_slots_with_drop_nan(
                slots,
                drop_nan_labels,
            ),
            success_criteria=_strategy_request_success_criteria(draft),
            auto_start=auto_start,
        )
    if draft.operation == "apply":
        return _start_confirmed_strategy_plan(
            runtime,
            repo,
            task,
            template_id="stored_strategy_apply",
            slots={
                "dataset_id": context.dataset_id,
                "strategy_id": draft.strategy_id,
            },
            auto_start=auto_start,
        )
    if draft.operation == "adopt":
        slots = _stored_strategy_slots(context, draft)
        slots["sample_design_ref"] = _latest_matching_strategy_sample_design_ref(
            runtime,
            task,
            context=context,
            drop_nan_labels=bool(drop_nan_labels),
            allow_native_risk_development=True,
        )
        return _start_confirmed_strategy_plan(
            runtime,
            repo,
            task,
            template_id="stored_strategy_adoption",
            slots=_strategy_slots_with_drop_nan(
                slots,
                drop_nan_labels,
            ),
            success_criteria=_strategy_request_success_criteria(draft),
            auto_start=auto_start,
        )

    if draft.operation == "develop":
        task = repo.update_strategy_input(
            task.id,
            _strategy_contract_from_draft(draft),
        )
        setup = _run_strategy_setup(
            runtime,
            repo,
            task,
            None,
            forced_intent=STRATEGY_INTENT_FULL_DEVELOPMENT,
        )
    elif draft.operation == "mine_rules":
        setup = _run_strategy_setup(
            runtime,
            repo,
            task,
            None,
            forced_intent=STRATEGY_INTENT_RULE_MINING,
        )
    elif draft.operation == "monitor":
        setup = _run_strategy_setup(
            runtime,
            repo,
            task,
            None,
            forced_intent=STRATEGY_INTENT_MONITORING,
        )
    else:  # guarded by _strategy_request_preflight
        raise StrategySetupError(f"strategy operation is not wired: {draft.operation}")
    if isinstance(setup, dict):
        return setup
    template_id, slots, start_kwargs = setup
    if template_id in {"strategy_development", "rule_strategy", "strategy_analysis"}:
        slots = dict(slots)
        slots["sample_design_ref"] = _latest_matching_strategy_sample_design_ref(
            runtime,
            task,
            context=context,
            drop_nan_labels=bool(drop_nan_labels),
            allow_native_risk_development=True,
        )
    if "success_criteria" not in start_kwargs:
        criteria = _strategy_request_success_criteria(draft)
        if criteria:
            start_kwargs = {**start_kwargs, "success_criteria": criteria}
    return _start_confirmed_strategy_plan(
        runtime,
        repo,
        task,
        template_id=template_id,
        slots=_strategy_slots_with_drop_nan(slots, drop_nan_labels),
        auto_start=auto_start,
        **start_kwargs,
    )

def _run_confirmed_strategy_request(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    pending: dict,
) -> dict:
    pending_repository = PendingStrategyRequestRepository(runtime.settings.db_path)
    request_id = str(pending.get("request_id") or "")
    payload_sha256 = str(pending.get("payload_sha256") or "")
    pending_record = pending_repository.get(task.id, request_id)
    if (
        pending_record is None
        or pending_record.status != "pending"
        or pending_record.payload_sha256 != payload_sha256
    ):
        return _strategy_request_clarification_response(
            repo,
            task,
            code="strategy_request_stale_confirmation",
            message="这份策略草案已被处理、篡改或失效，请重新描述并确认。",
        )
    persisted_payload = pending_record.validated_draft
    persisted_workflow = (
        persisted_payload.get("workflow")
        if isinstance(persisted_payload, Mapping)
        else None
    )
    persisted_sample_design = persisted_workflow in {
        "strategy_sample_design",
        "strategy_sample_design_v2",
    }
    preview = None
    preview_error = None
    try:
        preview = (
            _strategy_sample_design_dataset_preview(runtime, task)
            if persisted_sample_design
            else _strategy_dataset_preview(runtime, task)
        )
    except StrategySetupError as exc:
        preview_error = str(exc)
    expected_identity = pending_record.dataset_identity
    if expected_identity is not None and (
        preview is None
        or expected_identity != preview.identity
        or pending_record.target_col != preview.target_col
    ):
        _invalidate_pending_strategy_request(runtime, task, pending)
        return _strategy_request_clarification_response(
            repo,
            task,
            code="strategy_dataset_context_changed",
            message="策略样本或目标列已变化，请重新描述并确认策略请求。",
        )

    compilation = validate_strategy_request(
        pending_record.validated_draft,
        allowed_columns=_strategy_request_allowed_columns(preview),
        target_col=None if preview is None else preview.target_col,
        allow_legacy_replay=True,
    )
    if compilation.draft is None:
        _invalidate_pending_strategy_request(runtime, task, pending)
        return _strategy_request_clarification_response(
            repo,
            task,
            code="strategy_request_invalidated",
            message=compilation.clarification
            or "已确认的策略草案未通过重新校验，请重新描述。",
        )
    draft = compilation.draft
    preflight = _strategy_request_preflight(runtime, task, draft)
    if preflight is not None:
        _invalidate_pending_strategy_request(runtime, task, pending)
        code, message = preflight
        return _strategy_request_clarification_response(
            repo,
            task,
            code=code,
            message=message,
        )
    if _strategy_request_requires_dataset(draft) and preview is None:
        _invalidate_pending_strategy_request(runtime, task, pending)
        return _strategy_request_clarification_response(
            repo,
            task,
            code="strategy_dataset_context_required",
            message=preview_error or "当前策略操作需要一个任务内样本。",
        )
    if _strategy_request_requires_target(draft) and (
        preview is None or not preview.target_col
    ):
        _invalidate_pending_strategy_request(runtime, task, pending)
        return _strategy_request_clarification_response(
            repo,
            task,
            code="strategy_target_context_required",
            message="当前策略操作需要明确的二元目标列，请先在任务中指定 target_col。",
        )

    context = None
    if _strategy_request_requires_dataset(draft):
        is_sample_design = (
            isinstance(draft, StandardWorkflowRequestDraft)
            and draft.workflow
            in {"strategy_sample_design", "strategy_sample_design_v2"}
        )
        try:
            context = (
                _strategy_sample_design_dataset_context(runtime, task)
                if is_sample_design
                else _strategy_dataset_context(
                    runtime,
                    task,
                    require_target=_strategy_request_requires_target(draft),
                )
            )
        except StrategySetupError as exc:
            _invalidate_pending_strategy_request(runtime, task, pending)
            return _strategy_request_clarification_response(
                repo,
                task,
                code="strategy_dataset_context_required",
                message=str(exc),
            )
        if not _strategy_dataset_binding_matches(
            runtime,
            task,
            preview=preview,
            context=context,
            use_sample_design_workspace=is_sample_design,
        ):
            _invalidate_pending_strategy_request(runtime, task, pending)
            return _strategy_request_clarification_response(
                repo,
                task,
                code="strategy_dataset_context_changed",
                message=(
                    "策略样本在确认与计划创建之间发生变化；历史草案未消费，"
                    "请基于当前数据重新描述。"
                ),
            )
        if _strategy_request_requires_complete_labels(draft):
            try:
                n_total, n_nan = _strategy_target_nan_stats(runtime, context)
            except StrategySetupError as exc:
                _invalidate_pending_strategy_request(runtime, task, pending)
                return _strategy_request_clarification_response(
                    repo,
                    task,
                    code="strategy_target_labels_invalid",
                    message=str(exc),
                    fields=("target_col",),
                )
            confirmed_drop_nan = (
                isinstance(draft, StandardWorkflowRequestDraft)
                and draft.workflow_inputs.get("drop_nan_labels") is True
            )
            if n_nan and not confirmed_drop_nan:
                _invalidate_pending_strategy_request(runtime, task, pending)
                return _strategy_nan_label_clarification_response(
                    runtime,
                    repo,
                    task,
                    draft=draft,
                    context=context,
                    n_total=n_total,
                    n_nan=n_nan,
                )

    existing_plan_ids = frozenset(
        plan.id for plan in runtime.plan_repo.list_plans_for_task(task.id)
    )
    try:
        pending_repository.consume(
            task_id=task.id,
            request_id=request_id,
            expected_payload_sha256=payload_sha256,
        )
    except (
        PendingStrategyRequestConflictError,
        PendingStrategyRequestNotFoundError,
        ValueError,
    ):
        return _strategy_request_clarification_response(
            repo,
            task,
            code="strategy_request_stale_confirmation",
            message="这份策略草案已被其他请求处理或失效，请重新发起。",
        )
    repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content="已接收确认，正在按已校验口径创建受治理的策略计划。",
        metadata={
            "intent": "strategy_request_claimed",
            "request_id": request_id,
            "payload_sha256": payload_sha256,
            _STRATEGY_REQUEST_META_KEY: {
                "request_id": request_id,
                "payload_sha256": payload_sha256,
            },
        },
    )

    try:
        confirmed_drop_nan = (
            isinstance(draft, StandardWorkflowRequestDraft)
            and draft.workflow_inputs.get("drop_nan_labels") is True
        )
        return _run_validated_strategy_request(
            runtime,
            repo,
            task,
            draft,
            context=context,
            auto_start=False,
            drop_nan_labels=confirmed_drop_nan,
        )
    except _StrategyV2EvidenceSetupError as exc:
        return _strategy_request_clarification_response(
            repo,
            task,
            code=exc.code,
            message=str(exc),
        )
    except _StrategySampleDesignRequiredError as exc:
        try:
            pending_repository.release_after_failed_start(
                task_id=task.id,
                request_id=request_id,
                expected_payload_sha256=payload_sha256,
                existing_plan_ids=existing_plan_ids,
            )
        except (
            PendingStrategyRequestConflictError,
            PendingStrategyRequestNotFoundError,
            ValueError,
        ):
            pass
        return _strategy_request_clarification_response(
            repo,
            task,
            code="strategy_sample_design_required",
            message=str(exc),
        )
    except StrategySetupError as exc:
        return append_join_error(repo, task.id, str(exc))
    except DriverError:
        try:
            pending_repository.release_after_failed_start(
                task_id=task.id,
                request_id=request_id,
                expected_payload_sha256=payload_sha256,
                existing_plan_ids=existing_plan_ids,
            )
        except (
            PendingStrategyRequestConflictError,
            PendingStrategyRequestNotFoundError,
            ValueError,
        ):
            new_plans = [
                plan
                for plan in runtime.plan_repo.list_plans_for_task(task.id)
                if plan.id not in existing_plan_ids
            ]
            if new_plans:
                recovery_plan = new_plans[-1]
                repo.add_agent_message(
                    task.id,
                    role="assistant",
                    stage="chat",
                    content=(
                        "策略计划已经创建，但启动响应未完整返回；已保留现有计划，"
                        "请从该计划继续，旧确认不会重新创建计划。"
                    ),
                    metadata={
                        "intent": "strategy_request_plan_recovery",
                        "plan_id": recovery_plan.id,
                        "plan_status": recovery_plan.status.value,
                    },
                )
                return join_turn_response(repo, task.id)
        raise
    except Exception as exc:
        return append_join_error(repo, task.id, f"策略请求执行出错：{exc}")

def _start_confirmed_strategy_plan(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    *,
    template_id: str,
    slots: dict,
    success_criteria: list[dict] | None = None,
    auto_start: bool = False,
) -> dict:
    """Start a compiled Workflow, optionally auto-accepting its overview."""

    start_kwargs = {}
    if success_criteria:
        start_kwargs["success_criteria"] = success_criteria
    driver = _driver(runtime)
    start = driver.start(
        task_id=task.id,
        template_id=template_id,
        slots=slots,
        tier=runtime.tier,
        **start_kwargs,
    )
    append_driver_messages(repo, task, start, runtime=runtime)
    if auto_start:
        resumed = driver.resume(
            plan_id=start.plan_id,
            user_text="开始",
            confirmation_source=CONFIRMATION_SOURCE_AUTO,
        )
        append_driver_messages(repo, task, resumed, runtime=runtime)
    return join_turn_response(repo, task.id)

def _standard_workflow_request_preflight(
    runtime: DriverTurnRuntime,
    task: TaskRecord,
    draft: StandardWorkflowRequestDraft,
) -> tuple[str, str] | None:
    if draft.workflow == "strategy_project_context":
        # The exact persisted user message is request-local evidence.  It is
        # bound when the validated request starts, not reconstructed here from
        # whichever message happens to be latest during preflight.
        return None
    if draft.workflow == "strategy_sample_design":
        try:
            context = _strategy_sample_design_dataset_context(runtime, task)
            _strategy_sample_design_plan_slots(
                runtime,
                task,
                draft,
                context=context,
                drop_nan_labels=bool(
                    draft.workflow_inputs.get("drop_nan_labels", False)
                ),
            )
        except StrategySetupError as exc:
            return ("strategy_sample_design_workspace_required", str(exc))
        return None
    if draft.workflow == "strategy_sample_design_v2":
        try:
            context = _strategy_sample_design_dataset_context(runtime, task)
            _strategy_sample_design_v2_plan_slots(
                runtime,
                task,
                draft,
                context=context,
                drop_nan_labels=bool(
                    draft.workflow_inputs.get("drop_nan_labels", False)
                ),
            )
        except _StrategyV2EvidenceSetupError as exc:
            return (exc.code, str(exc))
        except StrategySetupError as exc:
            return ("strategy_sample_design_v2_workspace_required", str(exc))
        return None
    if draft.workflow == "strategy_model_evidence_v2":
        try:
            _strategy_model_evidence_v2_plan_slots(runtime, task)
        except _StrategyV2EvidenceSetupError as exc:
            return (exc.code, str(exc))
        except StrategySetupError as exc:
            return ("strategy_model_evidence_v2_binding_required", str(exc))
        return None
    if draft.workflow == "strategy_report_bundle_v2":
        # Bind exact refs only once, immediately before plan creation. A
        # separate preflight read would open a second selection window where
        # current source/report heads could silently rebind.
        return None
    if draft.workflow == "strategy_pool_validation":
        # Pool and exact mature SampleDesign V2 refs are authenticated together
        # once immediately before plan creation. The Tool repeats the exact-ref
        # checks under its artifact publication lock.
        return None
    if draft.workflow == "strategy_pool_apply":
        # Select and authenticate the exact current nonempty Pool only once,
        # immediately before plan creation. The Tool revalidates the CAS under
        # its writer lock before deriving any dataset.
        return None
    if draft.workflow == "strategy_pool_materialize":
        # Deep Pool/artifact/lineage authentication and all six immutable Tool
        # inputs are selected together once, immediately before plan creation.
        # The Tool repeats those exact-ref checks under its writer lock.
        return None
    if draft.workflow == "strategy_dsl_delivery":
        # Strategy and dataset refs are selected and authenticated together
        # exactly once immediately before plan creation.
        return None
    if draft.workflow == "strategy_impact_cube":
        # SampleDesign, Pool and optional current Strategy are selected and
        # authenticated together exactly once immediately before plan creation.
        return None
    if draft.workflow == "strategy_pool_stability":
        # The exact current Pool and latest complete SampleDesign V2 are bound
        # once during plan creation. The first step publishes the exact
        # ImpactCube and the second consumes only its direct output refs.
        return None
    if draft.workflow == "strategy_pool_impact":
        try:
            context = _strategy_pool_impact_dataset_context(runtime, task)
            _strategy_pool_impact_plan_slots(
                runtime,
                task,
                draft,
                context=context,
                drop_nan_labels=bool(
                    draft.workflow_inputs.get("drop_nan_labels", False)
                ),
            )
        except StrategySetupError as exc:
            return ("strategy_pool_impact_binding_required", str(exc))
        return None
    if draft.workflow == "automatic_tree_apply":
        try:
            context = _strategy_dataset_context(runtime, task, require_target=False)
            _automatic_tree_apply_slots(
                runtime,
                task_id=task.id,
                draft=draft,
                context=context,
            )
        except StrategySetupError as exc:
            return ("automatic_tree_apply_binding_required", str(exc))
        return None
    if draft.workflow == "automatic_tree_leaf_materialization":
        try:
            _automatic_tree_leaf_materialization_slots(
                runtime,
                task_id=task.id,
                draft=draft,
            )
        except StrategySetupError as exc:
            return ("automatic_tree_leaf_source_required", str(exc))
        return None
    if draft.workflow == "interactive_tree_split_search":
        try:
            _interactive_tree_split_search_plan_slots(
                runtime,
                task_id=task.id,
                draft=draft,
            )
        except StrategySetupError as exc:
            return (
                "interactive_tree_split_search_current_projection_required",
                str(exc),
            )
        return None
    if draft.workflow == "interactive_tree_revision":
        # Threshold edits are exposed by a manual tree projection, so reject a
        # stale/hidden split before a plan is created.  The Tool still repeats
        # full authentication under its writer lock before replay/persistence.
        # Existing prune requests intentionally retain their Tool-owned binding.
        if draft.workflow_inputs.get("operation") in {
            "adjust_split_threshold",
            "replace_split_feature",
        }:
            try:
                _interactive_tree_revision_plan_slots(
                    runtime,
                    task_id=task.id,
                    draft=draft,
                )
            except StrategySetupError as exc:
                return (
                    "interactive_tree_revision_current_projection_required",
                    str(exc),
                )
        return None
    if draft.workflow == "interactive_tree_auto_continuation":
        try:
            _interactive_tree_auto_continuation_plan_slots(
                runtime,
                task_id=task.id,
                draft=draft,
            )
        except StrategySetupError as exc:
            return (
                "interactive_tree_auto_continuation_search_required",
                str(exc),
            )
        return None
    if draft.workflow == "cross_matrix_cell_selection":
        try:
            _cross_matrix_cell_selection_slots(
                runtime,
                task_id=task.id,
                draft=draft,
            )
        except StrategySetupError as exc:
            return ("cross_matrix_cell_source_required", str(exc))
        return None
    if draft.workflow == "voting_candidate_search":
        try:
            _strategy_voting_candidate_search_plan_slots(runtime, task, draft)
        except StrategySetupError as exc:
            return ("strategy_voting_search_pool_binding_required", str(exc))
        return None
    if draft.workflow == "cross_matrix_candidate_search":
        try:
            _strategy_cross_candidate_search_plan_slots(runtime, task, draft)
        except StrategySetupError as exc:
            return ("strategy_cross_search_source_binding_required", str(exc))
        return None
    if draft.workflow == "cross_matrix_candidate_build_from_search":
        try:
            _strategy_cross_candidate_build_from_search_plan_slots(
                runtime,
                task,
                draft,
            )
        except StrategySetupError as exc:
            return ("strategy_cross_search_pair_binding_required", str(exc))
        return None
    if draft.workflow == "cross_rule_search":
        try:
            _strategy_cross_rule_search_plan_slots(runtime, task, draft)
        except StrategySetupError as exc:
            return ("strategy_cross_rule_source_binding_required", str(exc))
        return None
    if draft.workflow == "cross_rule_candidate_build_from_search":
        try:
            _strategy_cross_rule_candidate_build_plan_slots(
                runtime,
                task,
                draft,
            )
        except StrategySetupError as exc:
            return ("strategy_cross_rule_binding_required", str(exc))
        return None
    if draft.workflow == "voting_candidate_build_from_search":
        try:
            _strategy_voting_candidate_build_from_search_plan_slots(
                runtime,
                task,
                draft,
            )
        except StrategySetupError as exc:
            return ("strategy_voting_search_selection_binding_required", str(exc))
        return None
    if draft.workflow == "voting_candidate_build":
        try:
            _strategy_voting_candidate_plan_slots(runtime, task, draft)
        except StrategySetupError as exc:
            return ("strategy_voting_pool_binding_required", str(exc))
        return None
    if draft.workflow in _STRATEGY_POOL_WORKFLOWS:
        try:
            _strategy_pool_plan_slots(runtime, task, draft)
        except StrategySetupError as exc:
            return ("strategy_pool_binding_required", str(exc))
        return None
    return None

def _stored_strategy_request_preflight(
    runtime: DriverTurnRuntime,
    task: TaskRecord,
    draft: StrategyRequestDraft,
) -> tuple[str, str] | None:
    if not draft.strategy_id:
        return (
            "strategy_id_required",
            f"{draft.operation} 已有策略时必须说明 strategy_id。",
        )
    repository = StrategyRepository(runtime.settings.db_path)
    meta = repository.get_strategy_meta(draft.strategy_id)
    if meta is None or meta.get("task_id") != task.id:
        return (
            "strategy_not_owned_by_task",
            "没有在当前任务中找到该策略，不能跨任务读取或执行策略。",
        )
    if meta.get("strategy_type") != draft.strategy_type:
        return (
            "strategy_type_mismatch",
            "请求中的策略类型与已保存策略不一致，请按平台记录重新确认。",
        )

    if draft.baseline_strategy_id:
        baseline = repository.get_strategy_meta(draft.baseline_strategy_id)
        if baseline is None or baseline.get("task_id") != task.id:
            return (
                "strategy_baseline_not_owned_by_task",
                "没有在当前任务中找到基线策略，不能跨任务对比。",
            )
        if baseline.get("strategy_type") != draft.strategy_type:
            return (
                "strategy_baseline_type_mismatch",
                "候选策略与基线策略类型不一致，不能生成同口径对比。",
            )
        if draft.baseline_strategy_id == draft.strategy_id:
            return (
                "strategy_baseline_same_as_candidate",
                "候选策略和基线策略不能是同一个版本。",
            )
    if draft.operation == "compare" and not draft.baseline_strategy_id:
        return (
            "strategy_baseline_required",
            "策略对比必须说明 baseline_strategy_id。",
        )

    if draft.operation == "report":
        if any(
            value is not None
            for value in (
                draft.objective,
                draft.max_bad_rate,
                draft.min_approval_rate,
                draft.baseline_strategy_id,
                draft.adoption_reason,
                draft.profit,
                draft.economics_inputs,
            )
        ):
            return (
                "strategy_report_request_unused_fields",
                "已有策略报告只使用 strategy_id；其他业务字段不会被静默忽略，"
                "请删除后重试。",
            )
        return None

    if draft.operation == "apply":
        if any(
            value is not None
            for value in (
                draft.objective,
                draft.max_bad_rate,
                draft.min_approval_rate,
                draft.baseline_strategy_id,
                draft.adoption_reason,
                draft.profit,
                draft.economics_inputs,
            )
        ):
            return (
                "strategy_apply_request_unused_fields",
                "应用已有策略只使用 strategy_id 和任务内样本；目标、约束、基线、"
                "采纳理由及利润字段不会被静默忽略，请删除后重试。",
            )
        return None

    if draft.operation == "adopt":
        if not draft.adoption_reason:
            return (
                "strategy_adoption_reason_required",
                "采纳策略必须给出明确的人工采纳理由。",
            )
        if meta.get("status") != "draft":
            return (
                "strategy_draft_required",
                "只有 draft 或 validated 资产状态的策略可以进入本地采纳流程。",
            )
        if draft.baseline_strategy_id is not None:
            return (
                "strategy_adoption_unused_baseline",
                "采纳入口不会使用 baseline_strategy_id，请删除后重试。",
            )
        if (
            draft.strategy_type in {"limit", "pricing"}
            and draft.economics_inputs is None
        ):
            return (
                "strategy_adoption_economics_required",
                f"采纳 {draft.strategy_type} 策略需要完整 economics_inputs，"
                "以生成可审计的经济证据和类型化监控基线。",
            )
    elif draft.adoption_reason is not None:
        return (
            "strategy_request_unused_adoption_reason",
            "当前请求不是采纳操作，采纳理由不会被静默忽略。",
        )

    if draft.strategy_type not in {"approval", "reject"} and any(
        value is not None
        for value in (
            draft.objective,
            draft.max_bad_rate,
            draft.min_approval_rate,
            draft.profit,
        )
    ):
        return (
            "strategy_typed_business_contract_not_wired",
            f"{draft.strategy_type} 的目标、约束和经济参数需要类型专属 contract；"
            "当前不会忽略或套用审批口径。",
        )
    return None

def _latest_strategy_request_pending(conversation: list[dict]) -> dict | None:
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
    pending = (last_assistant.get("metadata") or {}).get(_STRATEGY_REQUEST_META_KEY)
    return pending if isinstance(pending, dict) else None

def _invalidate_pending_strategy_request(
    runtime: DriverTurnRuntime,
    task: TaskRecord,
    pending: dict,
) -> None:
    """Best-effort terminal transition for an obsolete opaque request ref."""

    try:
        PendingStrategyRequestRepository(runtime.settings.db_path).invalidate(
            task_id=task.id,
            request_id=str(pending.get("request_id") or ""),
            expected_payload_sha256=str(pending.get("payload_sha256") or ""),
        )
    except (
        PendingStrategyRequestConflictError,
        PendingStrategyRequestNotFoundError,
        ValueError,
    ):
        # The clarification path must remain safe under a concurrent confirm or
        # cancel. The winning transition is already audited by the repository.
        return

