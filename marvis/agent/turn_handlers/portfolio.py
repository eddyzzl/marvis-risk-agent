"""portfolio driver-turn handlers (executed into the package namespace by __init__.py)."""
from __future__ import annotations
from collections.abc import Mapping
from marvis.agent.plan_driver import CONFIRMATION_SOURCE_HUMAN, DriverError, is_confirm
from marvis.agent.portfolio_setup import PortfolioProposal, PortfolioSetupError, build_portfolio_proposal, build_states_gate_state, parse_states_reply, verify_portfolio_dataset_binding
from marvis.repositories.tasks import TaskRepository
from marvis.domain import TASK_TYPE_PORTFOLIO, TaskRecord
from marvis.orchestrator.contracts import PlanStatus
from marvis.repositories.plans import PlanRepository

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # names defined by sibling lanes; merged into one namespace at runtime
    from . import DriverTurnRuntime
    from . import _TurnHandlerSpec
    from . import _active_plan
    from . import _identity_display_text
    from . import _ingest_notice_text
    from . import _modeling_data_runtime
    from . import _run_driver_turn
    from . import _semantic_exact_gate_authorization
    from . import append_workflow_error
    from . import join_turn_response

def run_portfolio_driver_turn(
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
        _PORTFOLIO_SPEC,
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

def _portfolio_success_criteria(task: TaskRecord) -> list[dict] | None:
    """S3: optional deterministic criterion mirroring _strategy_success_criteria.
    task's optional portfolio_el_max (getattr-based -- no schema migration backs
    it) becomes a total_el ceiling final_review can evaluate; absent -> no
    criterion injected (same graceful default as strategy/modeling)."""
    el_max = getattr(task, "portfolio_el_max", None)
    if el_max is None:
        return None
    return [{"metric": "total_el", "max": float(el_max)}]

def _latest_portfolio_states(conversation: list[dict]) -> dict | None:
    for message in reversed(conversation):
        if message.get("role") != "assistant":
            continue
        meta = message.get("metadata") or {}
        if "portfolio_states" in meta:
            return meta["portfolio_states"]
    return None

def _request_portfolio_setup(
    repo: TaskRepository,
    task: TaskRecord,
) -> dict:
    repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content=(
            "开始组合分析前，请在组合口径表单中明确贷款id、快照月、逾期桶、"
            "余额/EAD、业务分群、损失态、LGD 和预测期限。平台不会从自由文本"
            "猜测这些业务语义；如需趋势分析，还须同时提供分数列和实验 ID。"
        ),
        metadata={
            "intent": "portfolio",
            "kind": "portfolio_setup_required",
            "required_fields": [
                "id_col",
                "snapshot_col",
                "bucket_col",
                "balance_col",
                "segment_col",
                "loss_state",
                "lgd",
                "horizon_months",
            ],
        },
    )
    return join_turn_response(repo, task.id)

def _begin_portfolio_setup(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    request: Mapping[str, object],
) -> dict:
    backend, registry = _modeling_data_runtime(runtime.settings)
    proposal = build_portfolio_proposal(
        registry,
        backend,
        task.id,
        task.source_dir,
        id_col=str(request.get("id_col") or "") or None,
        snapshot_col=str(request.get("snapshot_col") or "") or None,
        bucket_col=str(request.get("bucket_col") or "") or None,
        balance_col=str(request.get("balance_col") or "") or None,
        segment_col=str(request.get("segment_col") or "") or None,
        loss_state=str(request.get("loss_state") or "") or None,
        lgd=request.get("lgd"),
        horizon_months=request.get("horizon_months"),
        score_col=str(request.get("score_col") or "") or None,
        experiment_id=str(request.get("experiment_id") or "") or None,
    )
    notices = registry.consume_ingest_notices(task.id)
    states_text = " → ".join(f"`{state}`" for state in proposal.proposed_states)
    repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content=(
            f"开始组合分析:表现期表 `{proposal.dataset_name}`，贷款id `{proposal.id_col}`，"
            f"快照月 `{proposal.snapshot_col}`，逾期桶 `{proposal.bucket_col}`，"
            f"余额/EAD `{proposal.balance_col}`，业务分群 `{proposal.segment_col}`。\n"
            f"损失态 `{proposal.loss_state}`，LGD `{proposal.lgd:g}`，"
            f"预测期限 `{proposal.horizon_months}` 个月。\n"
            f"我按恶化程度排的桶顺序（由好到坏）：{states_text}。\n"
            "**桶的语义顺序机器不可猜，必须你确认**：无误时可以直接说明认可当前"
            "顺序；要改就按由好到坏顺序重列所有桶（逗号分隔）。"
            f"{_ingest_notice_text(notices)}"
        ),
        metadata={
            "portfolio_states": build_states_gate_state(proposal),
            "kind": "gate",
            "intent": "portfolio",
            "ingest_notices": notices,
        },
    )
    return join_turn_response(repo, task.id)

def _run_portfolio_setup(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    user_text: str | None,
    confirmation_source: str = CONFIRMATION_SOURCE_HUMAN,
) -> dict | tuple:
    conversation = repo.list_agent_messages(task.id)
    gate_state = _latest_portfolio_states(conversation)
    if gate_state is None:
        return _request_portfolio_setup(repo, task)

    text = str(user_text or "").strip()
    states = parse_states_reply(text, gate_state)
    semantic_authorization = None
    if (
        states is None
        and runtime.require_semantic_text_authorization
    ):
        proposed_states = [
            str(state) for state in gate_state.get("proposed_states") or []
        ]
        semantic_authorization = _semantic_exact_gate_authorization(
            runtime,
            text,
            gate_context=(
                "组合分析逾期桶顺序授权：用户已查看由好到坏的完整桶顺序；"
                "confirm 只表示明确、即时、无条件地接受当前完整顺序。任何重排"
                "仍必须逐字提交全部桶，LLM 不得猜测或改写顺序。"
            ),
            proposed_params={
                "dataset_content_hash": gate_state.get("dataset_content_hash"),
                "proposed_states": proposed_states,
            },
        )
        if semantic_authorization is not None:
            states = proposed_states
    if states is None:
        proposed = gate_state.get("proposed_states") or []
        states_text = " → ".join(f"`{state}`" for state in proposed)
        repo.add_agent_message(
            task.id,
            role="assistant",
            stage="chat",
            content=(
                "还没确认桶顺序。默认（由好到坏）："
                f"{states_text}。可以直接说明认可当前顺序，或按由好到坏重列所有桶"
                "（逗号分隔）；疑问、条件句和拒绝不会放行。"
            ),
            metadata={"portfolio_states": gate_state, "kind": "gate"},
        )
        return join_turn_response(repo, task.id)

    _, registry = _modeling_data_runtime(runtime.settings)
    verify_portfolio_dataset_binding(
        registry,
        task_id=task.id,
        dataset_id=str(gate_state["dataset_id"]),
        expected_content_hash=str(gate_state["dataset_content_hash"]),
    )

    proposal = PortfolioProposal(
        dataset_id=gate_state["dataset_id"],
        dataset_content_hash=gate_state["dataset_content_hash"],
        dataset_name="",
        id_col=gate_state["id_col"],
        snapshot_col=gate_state["snapshot_col"],
        bucket_col=gate_state["bucket_col"],
        proposed_states=list(states),
        balance_col=gate_state.get("balance_col"),
        segment_col=gate_state.get("segment_col"),
        loss_state=gate_state.get("loss_state"),
        lgd=gate_state.get("lgd"),
        horizon_months=gate_state.get("horizon_months"),
        score_col=gate_state.get("score_col"),
        experiment_id=gate_state.get("experiment_id"),
    )
    post_start_messages = []
    if semantic_authorization is not None:
        post_start_messages.append(
            {
                "role": "assistant",
                "stage": "chat",
                "content": "已通过独立语义复核，确认采用当前完整逾期桶顺序。",
                "metadata": {
                    "intent": "portfolio_semantic_authorization",
                    "display_in_timeline": False,
                    "dataset_content_hash": gate_state.get(
                        "dataset_content_hash"
                    ),
                    "proposed_states": list(states),
                    "semantic_authorization": semantic_authorization,
                },
            }
        )
    post_start_messages.append(
        {
            "role": "assistant",
            "stage": "chat",
            "content": (
                f"已确认桶顺序：{' → '.join(states)}。开始并行分析（流量/迁徙/细分"
                + ("/趋势" if proposal.experiment_id else "")
                + "），随后汇总确认。"
            ),
            "metadata": {"intent": "portfolio"},
        }
    )
    return (
        proposal.template_id,
        proposal.template_slots(states),
        {"_post_start_messages": post_start_messages},
    )

_PORTFOLIO_SPEC = _TurnHandlerSpec(
    intent="portfolio",
    setup_error_types=(PortfolioSetupError,),
    error_label="组合分析出错",
    run_setup=_run_portfolio_setup,
    format_user_display=_identity_display_text,
    success_criteria=_portfolio_success_criteria,
)

def is_portfolio_deterministic_turn(
    repo: TaskRepository,
    plan_repo: PlanRepository,
    task: TaskRecord,
    user_text: str | None,
) -> bool:
    """Allow only exact human replies at an existing Portfolio gate.

    A typed setup already bypasses the LLM.  This companion predicate keeps
    the immediately following state-order and plan confirmations usable in
    agent mode without turning arbitrary Portfolio chat into a manual-mode
    escape hatch.
    """

    if task.task_type != TASK_TYPE_PORTFOLIO:
        return False
    conversation = repo.list_agent_messages(task.id)
    last_assistant = next(
        (
            message
            for message in reversed(conversation)
            if message.get("role") == "assistant"
        ),
        None,
    )
    if last_assistant is None:
        return False
    metadata = last_assistant.get("metadata") or {}
    state_gate = metadata.get("portfolio_states")
    if isinstance(state_gate, dict):
        return parse_states_reply(user_text, state_gate) is not None
    if metadata.get("kind") == "portfolio_setup_required":
        return is_confirm(str(user_text or ""))

    active = _active_plan(plan_repo, task.id)
    if active is None or metadata.get("kind") not in {"gate", "plan_overview"}:
        return False
    status = PlanStatus(getattr(active.status, "value", active.status))
    return status in {PlanStatus.VALIDATED, PlanStatus.AWAITING_CONFIRM} and is_confirm(
        str(user_text or "")
    )

def _handle_structured_portfolio_request_turn(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    *,
    user_text: str | None,
    portfolio_request: Mapping[str, object],
) -> dict:
    """Start the human-owned portfolio setup contract without an LLM."""

    if task.task_type != TASK_TYPE_PORTFOLIO:
        raise DriverError("portfolio_request 只能用于 portfolio 类型任务。")
    if _active_plan(runtime.plan_repo, task.id) is not None:
        raise DriverError("当前组合分析任务已有进行中的计划，不能修改业务口径。")
    repo.add_agent_message(
        task.id,
        role="user",
        stage="chat",
        content=str(user_text or "").strip(),
        metadata={
            "intent": "portfolio_setup",
            "request_source": "manual_ui",
            "fields": sorted(portfolio_request),
        },
    )
    try:
        return _begin_portfolio_setup(
            runtime,
            repo,
            task,
            portfolio_request,
        )
    except PortfolioSetupError as exc:
        return append_workflow_error(
            repo,
            task,
            _PORTFOLIO_SPEC,
            exc,
            setup_error=True,
        )
