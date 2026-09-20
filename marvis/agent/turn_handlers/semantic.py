"""Semantic for governed Agent turns."""

from __future__ import annotations

from dataclasses import asdict
from marvis.agent.risk_analysis_setup import latest_risk_analysis_intake
from marvis.agent.semantic_intent import INTENT_ADHOC_CONFIRM
from marvis.agent.semantic_intent import INTENT_ADHOC_QUERY
from marvis.agent.semantic_intent import INTENT_ADHOC_REJECT
from marvis.agent.semantic_intent import INTENT_ADHOC_REVISE
from marvis.agent.semantic_intent import INTENT_CURRENT_WORKFLOW
from marvis.agent.semantic_intent import INTENT_DATASET_ANALYSIS
from marvis.agent.semantic_intent import INTENT_DATASET_EXPORT
from marvis.agent.semantic_intent import INTENT_DATASET_JOIN
from marvis.agent.semantic_intent import INTENT_DATASET_TRANSFORM
from marvis.agent.semantic_intent import INTENT_NONE
from marvis.agent.semantic_intent import INTENT_RISK_PROFITABILITY
from marvis.agent.semantic_intent import INTENT_RISK_STANDARD_VINTAGE
from marvis.agent.semantic_intent import INTENT_RISK_VTG_TERMINAL
from marvis.agent.semantic_intent import INTENT_STRATEGY_SAMPLE_BINDING
from marvis.agent.semantic_intent import INTENT_STRATEGY_WORKFLOW
from marvis.agent.strategy_setup import preview_strategy_dataset_context
from marvis.data.workspace import data_semantic_mapping_hash
from marvis.domain import TASK_TYPE_DATA_JOIN
from marvis.domain import TASK_TYPE_FEATURE_ANALYSIS
from marvis.domain import TASK_TYPE_MODELING
from marvis.domain import TASK_TYPE_STRATEGY
from marvis.domain import TASK_TYPE_VINTAGE
from marvis.domain import TaskRecord
from marvis.files import scan_data_workflow_dir
from marvis.orchestrator.contracts import plan_fingerprint
from marvis.repositories.data_workspace import DataWorkspaceRepository
from marvis.repositories.datasets import DatasetRepository
from marvis.repositories.tasks import TaskRepository
from pathlib import Path
import hashlib
import json
from . import adhoc as adhoc_lane
from . import c1_state as c1_state_lane
from . import contracts as contracts_lane
from . import data_context as data_context_lane
from . import shared as shared_lane


def _semantic_intent_clarification_response(
    repo: TaskRepository,
    task: TaskRecord,
    *,
    user_text: str,
    reason: str,
    pending_adhoc: dict | None = None,
) -> dict:
    """Persist a fail-closed Agent turn without changing workflow state."""

    repo.add_agent_message(
        task.id,
        role="user",
        stage="chat",
        content=user_text,
        metadata={"intent": "semantic_intent"},
    )
    repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content=(
            (
                "我还不能安全确定你是要执行、取消还是修改这份问数口径，"
                "因此没有执行工具，原口径仍等待确认。请直接说明要执行、取消，"
                "或给出新的分组、指标和筛选条件。"
            )
            if pending_adhoc is not None
            else (
                "我还不能安全确定这句话要进入哪条流程，因此没有创建计划、"
                "执行工具或改变当前状态。请直接说明要继续当前步骤，还是要做风险收益、"
                "Vintage/VTG、策略、数据处理、导出或问数。"
            )
        ),
        metadata={
            "intent": "semantic_intent",
            "kind": "clarification",
            "code": "semantic_intent_clarification",
            "reason": reason,
            **(
                {adhoc_lane._ADHOC_SPEC_META_KEY: dict(pending_adhoc)}
                if pending_adhoc is not None
                else {}
            ),
        },
    )
    return {
        "task_id": task.id,
        "status": "clarification_required",
        "code": "semantic_intent_clarification",
        "messages": repo.list_agent_messages(task.id),
    }


def _semantic_intent_state_snapshot(
    runtime: contracts_lane.DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
) -> str:
    """Hash every mutable identity that can change a top-level route's target.

    Agent messages and the active plan describe the conversational state, but
    current-data routes also depend on the live task row, DataWorkspace CAS
    revision, registered dataset identities, and pre-C1 source materials.
    Keeping all five surfaces in one digest makes an LLM decision disposable
    when any of them changes while the two independent semantic passes are in
    flight.

    ``DataWorkspaceRepository.get_or_default`` authenticates the active dataset
    owner/content hash against the dataset row before returning.  Dataset rows
    are included as well so a role, target, source identity, profile, or hash
    change that does not advance the workspace revision still invalidates the
    route.
    """

    conversation = repo.list_agent_messages(task.id)
    active = shared_lane._active_plan(runtime.plan_repo, task.id)
    live_task = repo.get_task(task.id)
    workspace = DataWorkspaceRepository(runtime.settings.db_path).get_or_default(
        task.id
    )
    datasets = DatasetRepository(runtime.settings.db_path).list_datasets(task.id)
    payload = {
        "task": asdict(live_task),
        "messages": [
            {
                "id": message.get("id"),
                "role": message.get("role"),
                "stage": message.get("stage"),
                "metadata": message.get("metadata") or {},
            }
            for message in conversation
        ],
        "active_plan": (
            None
            if active is None
            else {
                "id": active.id,
                "status": getattr(active.status, "value", active.status),
                "fingerprint": plan_fingerprint(active),
            }
        ),
        "data_workspace": {
            "schema_version": workspace.schema_version,
            "revision": workspace.revision,
            "active_dataset_id": workspace.active_dataset_id,
            "active_dataset_content_hash": workspace.active_dataset_content_hash,
            "analysis_generation": workspace.analysis_generation,
            "page": workspace.page,
            "selected_field": workspace.selected_field,
            "semantic_mapping_hash": data_semantic_mapping_hash(
                workspace.semantic_mapping
            ),
            "updated_at": workspace.updated_at,
        },
        "datasets": [
            asdict(dataset) for dataset in sorted(datasets, key=lambda item: item.id)
        ],
        "source_materials": _semantic_workflow_source_materials(live_task),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _semantic_workflow_source_materials(task: TaskRecord) -> list[dict[str, object]]:
    """Describe unregistered workflow inputs for TOCTOU invalidation.

    Before C1/setup, JOIN, Feature, and Modeling source tables are not dataset
    rows yet.  A text decision must be discarded if those materials change while
    the LLM is in flight.  Small inputs carry their content hash from
    ``scan_data_workflow_dir``; larger inputs retain path, size, and filesystem
    timestamps and are authenticated by the workflow setup before any plan runs.
    """

    if (
        task.task_type
        not in {
            TASK_TYPE_DATA_JOIN,
            TASK_TYPE_FEATURE_ANALYSIS,
            TASK_TYPE_MODELING,
            TASK_TYPE_STRATEGY,
        }
        or not str(task.source_dir or "").strip()
    ):
        return []
    source_dir = Path(task.source_dir).resolve()
    materials: list[dict[str, object]] = []
    for artifact in scan_data_workflow_dir(source_dir):
        path = Path(artifact.path).resolve()
        relative_path = path.relative_to(source_dir).as_posix()
        stat = path.stat()
        materials.append(
            {
                "relative_path": relative_path,
                "suffix": path.suffix.lower(),
                "size_bytes": int(stat.st_size),
                "sha256": artifact.sha256,
                "mtime_ns": int(stat.st_mtime_ns),
                "ctime_ns": int(stat.st_ctime_ns),
            }
        )
    return materials


def _semantic_join_material_context(
    runtime: contracts_lane.DriverTurnRuntime,
    task: TaskRecord,
) -> dict[str, object]:
    source_materials = _semantic_workflow_source_materials(task)
    source_names = [str(item["relative_path"]) for item in source_materials]
    registered = [
        dataset
        for dataset in DatasetRepository(runtime.settings.db_path).list_datasets(
            task.id
        )
        if str(dataset.role) in {"sample", "feature"}
    ]
    registered_names = [Path(dataset.source_path).name for dataset in registered]
    # Source names are the clearest initial context.  Once C1 has registered
    # more authenticated tables than remain in the source folder, those rows
    # are the available materials and must keep a continued JOIN route open.
    names = (
        registered_names if len(registered_names) > len(source_names) else source_names
    )
    return {
        "current_workflow": INTENT_DATASET_JOIN,
        "required_join_table_count": 2,
        "available_join_table_count": len(names),
        "available_join_tables": names,
    }


def _semantic_strategy_sample_binding_context(
    runtime: contracts_lane.DriverTurnRuntime,
    task: TaskRecord,
) -> dict[str, object]:
    """Expose one read-only, deterministic strategy binding candidate.

    The LLM decides whether the utterance authorizes this bounded operation.
    File and target identity remain platform facts: the route is unavailable
    unless the task has exactly one material and the read-only strategy setup
    can resolve exactly one target candidate without using the task's possibly
    stale ``target_col`` default.
    """

    source_materials = _semantic_workflow_source_materials(task)
    source_names = [str(item["relative_path"]) for item in source_materials]
    backend, registry = data_context_lane._modeling_data_runtime(runtime.settings)
    registered = [
        dataset
        for dataset in registry.list_for_task(task.id)
        if str(dataset.task_id) == task.id
        and dataset.role in {"sample", "strategy_sample"}
    ]
    registered_names: list[str] = []
    for dataset in registered:
        identity = registry.source_identity(dataset.id)
        registered_names.append(
            str((identity or {}).get("original_name") or Path(dataset.source_path).name)
        )
    names = source_names if source_names else registered_names
    workspace = DataWorkspaceRepository(runtime.settings.db_path).get_or_default(
        task.id
    )
    target_candidate: str | None = None
    preview_name: str | None = None
    if len(names) == 1 and workspace.active_dataset_id is None:
        preview = preview_strategy_dataset_context(
            registry,
            backend,
            task.id,
            task.source_dir,
            target_col=None,
        )
        preview_name = str(preview.dataset_name)
        target_candidate = str(preview.target_col or "").strip() or None
    preview_matches_unique_material = (
        len(names) == 1
        and preview_name is not None
        and (preview_name == names[0] or preview_name == Path(names[0]).name)
    )
    available = (
        len(names) == 1
        and preview_matches_unique_material
        and target_candidate is not None
        and workspace.active_dataset_id is None
    )
    return {
        "current_workflow": "strategy",
        "strategy_sample_binding_scope": (
            "authenticated_data_workspace_only_no_strategy_plan"
        ),
        "available_strategy_samples": names,
        "available_strategy_sample_count": len(names),
        "strategy_sample_target_candidate": (target_candidate if available else None),
        "strategy_sample_binding_available": available,
    }


def _semantic_intent_route_contract(
    runtime: contracts_lane.DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
) -> tuple[dict[str, object], tuple[str, ...], dict | None]:
    conversation = repo.list_agent_messages(task.id)
    pending_adhoc = adhoc_lane._latest_adhoc_pending(conversation)
    if pending_adhoc is not None:
        return (
            {"pending": "adhoc_query"},
            (
                INTENT_ADHOC_CONFIRM,
                INTENT_ADHOC_REJECT,
                INTENT_ADHOC_REVISE,
                INTENT_NONE,
            ),
            pending_adhoc,
        )

    risk_state = (
        latest_risk_analysis_intake(conversation)
        if task.task_type == TASK_TYPE_VINTAGE
        else None
    )
    risk_phase = str((risk_state or {}).get("phase") or "")
    if task.task_type == TASK_TYPE_VINTAGE and risk_phase != "ready":
        return (
            {"risk_setup_phase": risk_phase or "ask_goal"},
            (
                INTENT_RISK_PROFITABILITY,
                INTENT_RISK_VTG_TERMINAL,
                INTENT_RISK_STANDARD_VINTAGE,
                INTENT_CURRENT_WORKFLOW,
                INTENT_NONE,
            ),
            None,
        )

    allowed = [
        INTENT_ADHOC_QUERY,
        INTENT_DATASET_TRANSFORM,
        INTENT_DATASET_EXPORT,
        INTENT_DATASET_ANALYSIS,
        INTENT_CURRENT_WORKFLOW,
    ]
    context = {
        "risk_setup_phase": risk_phase,
        "has_ready_dataset": adhoc_lane._has_adhoc_dataset(
            runtime.settings,
            task.id,
        ),
    }
    if task.task_type == TASK_TYPE_DATA_JOIN:
        join_context = _semantic_join_material_context(runtime, task)
        join_context["join_exploration_phase"] = (
            "role_target_confirmation"
            if c1_state_lane._latest_c1_state(conversation) is not None
            else "role_discovery"
        )
        context.update(join_context)
        if int(join_context["available_join_table_count"]) >= int(
            join_context["required_join_table_count"]
        ):
            allowed.append(INTENT_DATASET_JOIN)
    if task.task_type == TASK_TYPE_STRATEGY:
        strategy_binding_context = _semantic_strategy_sample_binding_context(
            runtime,
            task,
        )
        context.update(strategy_binding_context)
        if bool(strategy_binding_context["strategy_sample_binding_available"]):
            allowed.append(INTENT_STRATEGY_SAMPLE_BINDING)
        allowed.append(INTENT_STRATEGY_WORKFLOW)
    if task.task_type == TASK_TYPE_VINTAGE:
        allowed.extend(
            (
                INTENT_RISK_PROFITABILITY,
                INTENT_RISK_VTG_TERMINAL,
                INTENT_RISK_STANDARD_VINTAGE,
            )
        )
    allowed.append(INTENT_NONE)
    return (context, tuple(allowed), None)
