"""typed_ui driver-turn handlers (executed into the package namespace by __init__.py)."""
from __future__ import annotations
import sqlite3
from marvis.agent.join_setup import AuthenticatedJoinSelection, C1TargetValidationError
from marvis.agent.plan_driver import DriverError, confirmation_is_explicitly_withheld
from marvis.agent.strategy_workflows import MANUAL_STANDARD_STRATEGY_WORKFLOWS
from marvis.data.registry import DatasetRegistry
from marvis.repositories.tasks import TaskRepository
from marvis.domain import TaskRecord
from marvis.orchestrator.contracts import Plan, PlanStatus, StepStatus, plan_fingerprint, plan_step_confirmation_fingerprint
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.sample_design_binding import StrategySampleDesignRef

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # names defined by sibling lanes; merged into one namespace at runtime
    from . import DriverTurnRuntime
    from . import _STRATEGY_SAMPLE_BOUND_TOOLS
    from . import _TERMINAL_PLAN_STATUS_VALUES
    from . import _TurnHandlerSpec
    from . import append_driver_messages

def _validate_typed_ui_action_target(
    plan: Plan | None,
    *,
    ui_action: str | None,
    user_text: str | None,
    expected_plan_id: str | None,
    expected_step_id: str | None,
    expected_plan_status: str | None,
    expected_plan_revision: int | None,
    expected_plan_fingerprint: str | None,
    expected_step_fingerprint: str | None,
) -> None:
    """Fail closed when a rendered authorization control is stale.

    This executes inside the task's driver-job lock and before any success audit
    message is persisted. A plan id alone is insufficient for ``start_plan``:
    the same plan may already have advanced to a per-step gate in another tab.
    """

    plan_actions = {
        "start_plan",
        "confirm_dedup",
        "apply_join_keys",
        "exclude_join_feature",
        "confirm_features",
        "adjust_screen_thresholds",
        "confirm_feature_binning",
        "apply_modeling_setup",
        "confirm_adoption",
        "confirm_gate",
    }
    if ui_action not in plan_actions:
        return
    if confirmation_is_explicitly_withheld(user_text or ""):
        raise DriverError("界面操作与停止或暂缓指令冲突，请刷新后重新选择。")
    # The action enum is the authorization signal.  ``user_text`` is only the
    # localized audit/display copy and may legitimately describe an adjustment
    # (for example “重新诊断拼接键” or “调整建模规格…”).  Requiring a magic
    # confirmation word here would make genuine rendered controls unusable.
    rendered_plan_id = str(expected_plan_id or "").strip()
    if plan is None or not rendered_plan_id or rendered_plan_id != plan.id:
        raise DriverError("该操作对应的计划已变化，请刷新页面后重试。")
    if (
        expected_plan_status is None
        or expected_plan_revision is None
        or expected_plan_fingerprint is None
    ):
        raise DriverError("该操作缺少完整的计划快照，请刷新页面后重试。")
    plan_status = PlanStatus(getattr(plan.status, "value", plan.status))
    if str(expected_plan_status) != plan_status.value:
        raise DriverError("该操作对应的计划状态已变化，请刷新页面后重试。")
    if int(expected_plan_revision) != int(plan.replan_count):
        raise DriverError("该操作对应的计划版本已变化，请刷新页面后重试。")
    if str(expected_plan_fingerprint) != plan_fingerprint(plan):
        raise DriverError("该操作对应的计划内容已变化，请刷新页面后重试。")
    if ui_action == "start_plan":
        if plan_status != PlanStatus.VALIDATED:
            raise DriverError("该开始按钮已过期，请刷新页面后操作当前步骤。")
        return
    if plan_status != PlanStatus.AWAITING_CONFIRM:
        raise DriverError("该确认按钮已过期，请刷新页面后操作当前步骤。")
    rendered_step_id = str(expected_step_id or "").strip()
    current_gate = next(
        (
            step
            for step in plan.steps
            if getattr(step.status, "value", step.status)
            == StepStatus.AWAITING_CONFIRM.value
        ),
        None,
    )
    if (
        current_gate is None
        or not rendered_step_id
        or rendered_step_id != current_gate.id
    ):
        raise DriverError("该确认按钮对应的步骤已变化，请刷新页面后重试。")
    if expected_step_fingerprint is None:
        raise DriverError("该操作缺少完整的步骤快照，请刷新页面后重试。")
    if str(expected_step_fingerprint) != plan_step_confirmation_fingerprint(
        current_gate,
        confirmed=False,
    ):
        raise DriverError("该确认按钮对应的步骤内容已变化，请刷新页面后重试。")
    gate_tool = current_gate.tool_ref.tool
    dependency_tools = {
        step.tool_ref.tool
        for step in plan.steps
        if step.id in set(current_gate.depends_on or [])
    }
    action_matches_gate = {
        "confirm_dedup": (
            gate_tool == "execute_join" and "confirm_join" in dependency_tools
        ),
        "apply_join_keys": (
            gate_tool == "execute_join" and "propose_join" in dependency_tools
        ),
        "exclude_join_feature": (
            gate_tool == "execute_join" and "propose_join" in dependency_tools
        ),
        "confirm_features": "screen_features" in dependency_tools,
        "adjust_screen_thresholds": "screen_features" in dependency_tools,
        "confirm_feature_binning": gate_tool == "analyze_feature_bins",
        "apply_modeling_setup": (
            gate_tool == "screen_features"
            and "choose_modeling_spec" in dependency_tools
        ),
        "confirm_adoption": "adoption_reason" in (current_gate.inputs or {}),
    }
    if ui_action in action_matches_gate and not action_matches_gate[ui_action]:
        raise DriverError("该界面操作与当前待确认步骤不匹配，请刷新页面后重试。")

def _append_successful_ui_action_messages(
    spec: _TurnHandlerSpec,
    repo: TaskRepository,
    task: TaskRecord,
    *,
    user_text: str | None,
    ui_action: str | None,
    expected_plan_id: str | None,
    expected_step_id: str | None,
    conn: sqlite3.Connection | None = None,
) -> None:
    """Persist UI authorization evidence only after the command succeeds."""

    if user_text is None or not ui_action:
        return
    message_metadata: dict[str, object] = {
        "intent": spec.intent,
        "ui_action": ui_action,
        "display_in_timeline": False,
    }
    if expected_plan_id:
        message_metadata["expected_plan_id"] = expected_plan_id
    if expected_step_id:
        message_metadata["expected_step_id"] = expected_step_id
    def add_message(**kwargs) -> None:
        if conn is None:
            repo.add_agent_message(task.id, **kwargs)
        else:
            repo.add_agent_message_on_connection(conn, task.id, **kwargs)
    add_message(
        role="user",
        stage="chat",
        content=spec.format_user_display(user_text),
        metadata=message_metadata,
    )
    acknowledgements = {
        "confirm_roles": "收到角色与目标列确认，开始生成执行计划。",
        "confirm_dedup": "收到去重策略确认，开始继续拼接。",
        "apply_join_keys": "收到拼接键选择，正在重新诊断拼接方案。",
        "exclude_join_feature": "收到特征表排除调整，正在重新诊断拼接。",
        "confirm_features": "收到特征选择确认，开始执行下一步。",
        "adjust_screen_thresholds": "收到筛选阈值调整，正在重新计算。",
        "confirm_feature_binning": "收到分箱选择，开始生成分箱结果和特征分析报告。",
        "apply_modeling_setup": "收到建模设置，开始重算后续步骤。",
        "confirm_adoption": "收到采纳确认，开始绑定理由并生成审计记录。",
        "confirm_gate": "收到确认，开始执行下一步。",
        "start_plan": "收到确认，开始按计划执行。",
    }
    acknowledgement_metadata: dict[str, object] = {
        "intent": "ui_action_ack",
        "ui_action": ui_action,
    }
    if expected_plan_id:
        acknowledgement_metadata["expected_plan_id"] = expected_plan_id
    if expected_step_id:
        acknowledgement_metadata["expected_step_id"] = expected_step_id
    add_message(
        role="assistant",
        stage="chat",
        content=acknowledgements.get(ui_action, "收到确认，开始执行下一步。"),
        metadata=acknowledgement_metadata,
    )

def _terminate_stale_strategy_sample_plan(
    spec: _TurnHandlerSpec,
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    plan: Plan,
) -> dict | None:
    """Fail closed before resuming pre-sample-binding strategy plans.

    V2 plans are serialized, so a plan created before sample-design binding can
    outlive a deployment and otherwise resume directly into a data-reading Tool.
    Only unfinished sample-bound steps are migration-sensitive: completed
    historical evidence remains readable, while newly compiled plans carry an
    exact authenticated development-partition reference and resume unchanged.
    """

    if spec.intent != "strategy":
        return None
    stale_steps = _stale_strategy_sample_steps(plan)
    if not stale_steps:
        return None

    current_status = PlanStatus(getattr(plan.status, "value", plan.status))
    if current_status in {
        PlanStatus.DRAFT,
        PlanStatus.VALIDATED,
        PlanStatus.RUNNING,
        PlanStatus.REVIEW,
    }:
        terminal_status = PlanStatus.FAILED
    elif current_status in {
        PlanStatus.CONFIRMED,
        PlanStatus.AWAITING_CONFIRM,
    }:
        # These states cannot legally transition directly to FAILED. CANCELLED
        # is their governed terminal path and prevents any Tool invocation.
        terminal_status = PlanStatus.CANCELLED
    else:
        return None

    runtime.plan_repo.set_plan_status(plan.id, terminal_status)
    stale_step_payload = [
        {
            "step_id": step.id,
            "tool": step.tool_ref.tool,
            "step_status": getattr(step.status, "value", step.status),
        }
        for step in stale_steps
    ]
    runtime.plan_repo.write_audit(
        kind="strategy.plan.sample_design_stale",
        target_ref=plan.id,
        outcome="blocked",
        detail={
            "task_id": task.id,
            "template_id": plan.template_id,
            "from_status": current_status.value,
            "to_status": terminal_status.value,
            "clarification_code": "strategy_plan_sample_design_stale",
            "stale_steps": stale_step_payload,
        },
    )
    message = (
        "该策略计划由旧版本创建，未完成的数据分析或回测步骤没有绑定当前成熟样本设计的"
        "精确 sample_design_ref。平台已安全终止旧计划，且没有调用任何分析工具。"
        "请先确认当前成熟样本设计，再基于该设计重新发起策略请求；平台会重建计划。"
    )
    repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content=message,
        metadata={
            "intent": "strategy",
            "kind": "clarification",
            "code": "strategy_plan_sample_design_stale",
            "plan_id": plan.id,
            "template_id": plan.template_id,
            "plan_status": terminal_status.value,
            "stale_steps": stale_step_payload,
        },
    )
    return {
        "task_id": task.id,
        "status": "clarification_required",
        "code": "strategy_plan_sample_design_stale",
        "plan_id": plan.id,
        "plan_status": terminal_status.value,
        "messages": repo.list_agent_messages(task.id),
    }

def _stale_strategy_sample_steps(plan: Plan) -> list:
    status = getattr(plan.status, "value", plan.status)
    if status in _TERMINAL_PLAN_STATUS_VALUES:
        return []
    stale_steps = []
    for step in plan.steps:
        step_status = getattr(step.status, "value", step.status)
        if step_status in {StepStatus.DONE.value, StepStatus.SKIPPED.value}:
            continue
        if (
            step.tool_ref.plugin != "strategy"
            or step.tool_ref.tool not in _STRATEGY_SAMPLE_BOUND_TOOLS
        ):
            continue
        try:
            StrategySampleDesignRef.from_value(
                step.inputs.get("sample_design_ref")
            )
        except StrategyError:
            stale_steps.append(step)
    return stale_steps

def _append_spec_messages(
    repo: TaskRepository,
    task: TaskRecord,
    turn,
    runtime: DriverTurnRuntime,
) -> None:
    append_driver_messages(repo, task, turn, runtime=runtime)

def _identity_display_text(user_text: str) -> str:
    return user_text

def _validated_authenticated_c1_target(
    registry: DatasetRegistry,
    selection: AuthenticatedJoinSelection,
    target_col: object,
) -> str | None:
    """Bind a submitted target to the schema of the authenticated anchor bytes."""

    normalized = str(target_col or "").strip() or None
    if normalized is None:
        return None
    anchor_columns = set(
        registry.authenticated_binding_column_names(selection.anchor)
    )
    if normalized in anchor_columns:
        return normalized
    feature_only = any(
        normalized
        in set(registry.authenticated_binding_column_names(binding))
        for binding in selection.features
    )
    if feature_only:
        raise C1TargetValidationError(
            f"目标列 `{normalized}` 只存在于特征表；目标列必须来自当前样本主表。"
        )
    raise C1TargetValidationError(
        f"目标列 `{normalized}` 不存在于当前样本主表；请重新选择。"
    )

_TYPED_EVALUATION_OPERATIONS = frozenset({"analyze", "backtest"})

_STORED_EVALUATION_OPERATIONS = frozenset({"analyze", "backtest", "compare"})

_MANUAL_STRATEGY_WORKFLOWS = frozenset(MANUAL_STANDARD_STRATEGY_WORKFLOWS)

def _auto_decision_content(decision: dict) -> str:
    reason = str(decision.get("reason") or "").strip() or "自动决策已生成。"
    action = decision.get("action")
    if action == "clarify" and decision.get("clarifying_question"):
        return f"🤖 {reason}\n\n需要确认:{decision['clarifying_question']}"
    if action == "replan" and decision.get("replan_goal"):
        return f"🤖 {reason}\n\n重规划目标:{decision['replan_goal']}"
    # LT-11 (B.3): when AUTO auto-confirms a low-risk gate, append the "why safe"
    # rationale (_apply_safety_policy attached it because no risk flag / wide reset
    # fired) so the auto-confirm explains itself. A halt already cites the specific
    # risk_flag code in its reason (from _gate_risk_reason), so no extra line there.
    rationale = str(decision.get("safety_rationale") or "").strip()
    if action == "confirm" and rationale:
        return f"🤖 {reason}\n\n为何可自动确认:{rationale}"
    return f"🤖 {reason}"

