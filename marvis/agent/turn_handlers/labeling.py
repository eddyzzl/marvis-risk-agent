"""labeling driver-turn handlers (executed into the package namespace by __init__.py)."""
from __future__ import annotations
from collections.abc import Mapping
import hmac
import sqlite3
from marvis.agent.plan_driver import CONFIRMATION_SOURCE_HUMAN, DriverError, is_confirm
from marvis.data.errors import DatasetContentDriftError
from marvis.repositories.tasks import TaskRepository
from marvis.domain import TASK_TYPE_DATA_JOIN, TaskRecord
from marvis.packs.labeling.contracts import LabelingContractError, LabelingRequest, build_labeling_proposal
from marvis.repositories.plans import PlanRepository
from marvis.repositories.data_workspace import DataWorkspaceRepository

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # names defined by sibling lanes; merged into one namespace at runtime
    from . import DriverTurnRuntime
    from . import _active_plan
    from . import _driver
    from . import _modeling_data_runtime
    from . import _semantic_exact_gate_authorization
    from . import join_turn_response

def _handle_structured_labeling_request_turn(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    *,
    user_text: str | None,
    labeling_request: Mapping[str, object],
) -> dict:
    """Build a read-only label proposal; never create or execute a plan."""

    if task.task_type != TASK_TYPE_DATA_JOIN:
        raise DriverError("labeling_request 只能用于 data_join 类型任务。")
    if _active_plan(runtime.plan_repo, task.id) is not None:
        raise DriverError("当前数据处理任务已有进行中的计划，不能修改标签构造口径。")
    try:
        contract = LabelingRequest(**dict(labeling_request))
    except (LabelingContractError, TypeError) as exc:
        return _labeling_clarification_response(
            repo,
            task,
            code="labeling_request_invalid",
            content=f"标签构造口径无效：{exc}",
        )

    repo.add_agent_message(
        task.id,
        role="user",
        stage="chat",
        content=str(user_text or "").strip(),
        metadata={
            "intent": "labeling_setup",
            "request_source": "manual_ui",
            "proposal_hash": contract.contract_hash,
            "fields": sorted(labeling_request),
        },
    )
    backend, registry = _modeling_data_runtime(runtime.settings)
    workspace = DataWorkspaceRepository(runtime.settings.db_path).get_or_default(
        task.id
    )
    try:
        proposal = build_labeling_proposal(
            registry,
            backend,
            workspace,
            task_id=task.id,
            request=contract,
        )
    except (LabelingContractError, DatasetContentDriftError, KeyError) as exc:
        return _labeling_clarification_response(
            repo,
            task,
            code="labeling_request_invalid",
            content=f"标签构造提案未通过源数据校验：{exc}",
            proposal_hash=contract.contract_hash,
        )

    maturity = proposal.maturity
    maturity_text = (
        "全部 cohort 已成熟"
        if maturity["all_matured"]
        else (
            f"{len(maturity['immature_cohorts'])} 个 cohort 尚未成熟；确认后这些贷款"
            "会保留为 NaN 标签，并继续受下游空标签门控制"
        )
    )
    repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content=(
            "标签构造提案已生成，**尚未创建计划，也未写入数据**。\n"
            f"- 数据集：`{proposal.source_dataset_name}`（{proposal.rows_at_as_of} 行纳入 "
            f"as-of，{proposal.rows_excluded_after_as_of} 行晚于截止日而排除）\n"
            f"- 截止日：`{contract.as_of_date}`\n"
            f"- 观察期 / 表现期 / 定坏时点：MOB {contract.observation_window} / "
            f"{contract.performance_window} / {contract.at_mob}\n"
            f"- 定坏规则：`{proposal.rule_summary}` → `{contract.target_col}`\n"
            f"- 成熟度：{maturity_text}\n"
            "请仅在上述口径和成熟度处理都无误时明确授权继续（可以直接说明你认可"
            "当前完整提案）。修改任何字段时，请重新提交完整结构化口径；平台不会"
            "从回复中猜测或拼接字段。"
        ),
        metadata={
            "intent": "labeling_setup",
            "kind": "labeling_preplan_confirmation",
            "labeling_proposal": proposal.to_gate_state(),
        },
    )
    return join_turn_response(repo, task.id)

def _maybe_handle_labeling_preplan_confirmation_turn(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    *,
    user_text: str | None,
    confirmation_source: str,
) -> dict | None:
    """Require one independently reviewed human turn before a label plan exists."""

    if task.task_type != TASK_TYPE_DATA_JOIN:
        return None
    if _active_plan(runtime.plan_repo, task.id) is not None:
        return None
    proposal_state = _latest_open_labeling_proposal(
        repo.list_agent_messages(task.id)
    )
    if proposal_state is None:
        return None

    text = str(user_text or "").strip()
    proposal_hash = str(proposal_state.get("proposal_hash") or "")
    semantic_authorization = None
    deterministically_confirmed = (
        confirmation_source == CONFIRMATION_SOURCE_HUMAN and is_confirm(text)
    )
    if (
        not deterministically_confirmed
        and confirmation_source == CONFIRMATION_SOURCE_HUMAN
        and runtime.require_semantic_text_authorization
    ):
        semantic_authorization = _semantic_exact_gate_authorization(
            runtime,
            text,
            gate_context=(
                "标签构造计划创建授权：用户已查看当前完整标签口径、成熟度处理和"
                "提案内容；confirm 只表示用户明确、即时、无条件地授权按该完整"
                "提案创建计划，不允许从普通文本修改任何字段。"
            ),
            proposed_params={"proposal_hash": proposal_hash},
        )
    if not deterministically_confirmed and semantic_authorization is None:
        repo.add_agent_message(
            task.id,
            role="user",
            stage="chat",
            content=text,
            metadata={
                "intent": "labeling_preplan_confirmation",
                "confirmation_source": confirmation_source,
                "proposal_hash": proposal_hash,
            },
        )
        return _labeling_clarification_response(
            repo,
            task,
            code="labeling_human_confirmation_required",
            content=(
                "标签构造会定义建模目标，必须由人工对当前完整提案作无歧义授权。"
                "你可以直接说明认可当前方案；疑问、条件句、拒绝或修改要求都不会"
                "放行。如需修改，请重新提交完整结构化口径。"
            ),
            proposal_hash=proposal_hash,
        )

    try:
        request_payload = proposal_state.get("request")
        if not isinstance(request_payload, dict):
            raise LabelingContractError("persisted proposal request is unavailable")
        contract = LabelingRequest(**request_payload)
        if not hmac.compare_digest(contract.contract_hash, proposal_hash):
            raise LabelingContractError("persisted proposal hash is inconsistent")
        backend, registry = _modeling_data_runtime(runtime.settings)
        workspace = DataWorkspaceRepository(
            runtime.settings.db_path
        ).get_or_default(task.id)
        proposal = build_labeling_proposal(
            registry,
            backend,
            workspace,
            task_id=task.id,
            request=contract,
        )
    except (LabelingContractError, DatasetContentDriftError, KeyError) as exc:
        return _labeling_clarification_response(
            repo,
            task,
            code="labeling_proposal_stale",
            content=(
                "标签构造提案绑定的源数据或 DataWorkspace 已变化，原确认已拒绝；"
                f"请刷新并重新提交完整口径。校验信息：{exc}"
            ),
            proposal_hash=proposal_hash,
            resolution="stale",
        )

    def persist_confirmation_and_overview(
        conn: sqlite3.Connection,
        turn,
    ) -> None:
        repo.add_agent_message_on_connection(
            conn,
            task.id,
            role="user",
            stage="chat",
            content=text,
            metadata={
                "intent": "labeling_preplan_confirmation",
                "confirmation_source": CONFIRMATION_SOURCE_HUMAN,
                "proposal_hash": proposal_hash,
                "semantic_authorized": semantic_authorization is not None,
            },
        )
        if semantic_authorization is not None:
            repo.add_agent_message_on_connection(
                conn,
                task.id,
                role="assistant",
                stage="chat",
                content="已通过独立语义复核，确认采用当前完整标签口径提案。",
                metadata={
                    "intent": "labeling_semantic_authorization",
                    "display_in_timeline": False,
                    "proposal_hash": proposal_hash,
                    "semantic_authorization": semantic_authorization,
                },
            )
        repo.add_agent_message_on_connection(
            conn,
            task.id,
            role="assistant",
            stage="chat",
            content="已记录人工标签口径确认，正在生成可审查的执行计划。",
            metadata={
                "intent": "labeling_preplan_confirmation",
                "proposal_hash": proposal_hash,
                "labeling_proposal_resolution": "confirmed",
            },
        )
        for message in turn.messages:
            repo.add_agent_message_on_connection(
                conn,
                task.id,
                role="assistant",
                stage="chat",
                content=message.content,
                metadata=dict(message.metadata),
            )

    _driver(runtime).start(
        task_id=task.id,
        template_id="label_construction",
        slots=proposal.to_template_slots(
            confirm_immature_cohorts=bool(
                proposal.maturity["immature_cohorts"]
            ),
        ),
        tier=runtime.tier,
        _persist_start_turn=persist_confirmation_and_overview,
    )
    return join_turn_response(repo, task.id)

def _latest_open_labeling_proposal(
    conversation: list[dict],
) -> dict[str, object] | None:
    for message in reversed(conversation):
        metadata = message.get("metadata") or {}
        if metadata.get("labeling_proposal_resolution"):
            return None
        proposal = metadata.get("labeling_proposal")
        if (
            metadata.get("kind") == "labeling_preplan_confirmation"
            and isinstance(proposal, dict)
        ):
            return proposal
    return None

def is_labeling_deterministic_turn(
    repo: TaskRepository,
    plan_repo: PlanRepository,
    task: TaskRecord,
    user_text: str | None,
) -> bool:
    """Bypass LLM only for exact deterministic label-flow confirmations."""

    if task.task_type != TASK_TYPE_DATA_JOIN:
        return False
    if _latest_open_labeling_proposal(repo.list_agent_messages(task.id)) is not None:
        return is_confirm(str(user_text or ""))
    # Once the plan exists, ordinary Agent text must go through the same
    # semantic route/review as every other live plan gate.  Browser controls
    # remain deterministic through their typed, snapshot-bound ``ui_action``.
    return False

def _labeling_clarification_response(
    repo: TaskRepository,
    task: TaskRecord,
    *,
    code: str,
    content: str,
    proposal_hash: str | None = None,
    resolution: str | None = None,
) -> dict:
    metadata: dict[str, object] = {
        "intent": "labeling_setup",
        "kind": "clarification",
        "code": code,
    }
    if proposal_hash:
        metadata["proposal_hash"] = proposal_hash
    if resolution:
        metadata["labeling_proposal_resolution"] = resolution
    repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content=content,
        metadata=metadata,
    )
    return {
        "task_id": task.id,
        "status": "clarification_required",
        "messages": repo.list_agent_messages(task.id),
    }

