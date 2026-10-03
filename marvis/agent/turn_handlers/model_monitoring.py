"""A monitoring proposal uses the existing plan overview as its sole human gate."""
from __future__ import annotations

from marvis.agent.monitoring_setup import ModelMonitoringSetupRequest, prepare_model_monitoring
from marvis.agent.plan_driver import DriverError
from marvis.data.errors import DataLayerError
from marvis.domain import TASK_TYPE_MODELING
from marvis.orchestrator.evidence import payload_hash
from . import responses as responses_lane
from . import shared as shared_lane


def handle_model_monitoring_request(runtime, repo, task, *, user_text, model_monitoring_request):
    if task.task_type != TASK_TYPE_MODELING:
        raise DriverError("model_monitoring_request 只能用于 modeling 类型任务。")
    if shared_lane._active_plan(runtime.plan_repo, task.id) is not None:
        raise DriverError("当前任务已有进行中的计划，请完成或取消后再创建模型监控。")
    contract = ModelMonitoringSetupRequest.model_validate(model_monitoring_request)
    try:
        proposal = prepare_model_monitoring(runtime.settings, task.id, contract)
    except (ValueError, KeyError, DataLayerError, OSError) as exc:
        messages = {
            "monitoring_binding_authentication_failed": "模型或数据的来源证明无法验证。请重新导入数据；早期模型请重新训练并选定后再监控。",
            "monitoring_selected_task_experiment_required": "所选模型已变化或不属于本任务，请刷新后重新选择。",
            "monitoring_baseline_required": "所选模型缺少训练期分布基线，请重新训练并选定后再监控。",
            "monitoring_binary_baseline_required": "当前监控需要有训练期分布基线的二分类模型，请选择适用模型。",
        }
        raise DriverError(messages.get(str(exc), str(exc))) from exc
    proposal_hash = payload_hash(proposal)
    label_text = (
        "未提供标签，仅运行漂移检查，KS/AUC 明确不可用。"
        if contract.target_col is None else
        f"使用声明标签列 {contract.target_col}；成熟度未知，不能据此宣称真实业务效果达标。"
    )
    explanation = (
        f"模型监控提案：已选实验 {contract.experiment_id}，新数据 {contract.dataset_id}。\n"
        "模型、训练基线和新数据已绑定到本次计划。阈值是平台默认技术阈值，"
        "不能替代业务合同目标。" + label_text + "\n确认下方计划后开始打分与监控。"
    )

    def persist(conn, turn):
        # The same transaction that publishes the overview checks that no
        # concurrent workspace edit changed the proposal's displayed sources.
        try:
            current = prepare_model_monitoring(runtime.settings, task.id, contract)
        except (ValueError, KeyError, DataLayerError, OSError) as exc:
            raise DriverError("监控提案来源已变化，请刷新后重建计划。") from exc
        if payload_hash(current) != proposal_hash:
            raise DriverError("监控提案来源已变化，请刷新后重建计划。")
        repo.add_agent_message_on_connection(
            conn, task.id, role="user", stage="chat", content=str(user_text),
            metadata={"intent": "model_monitoring", "request_source": "typed_user_request"},
        )
        for index, message in enumerate(turn.messages):
            repo.add_agent_message_on_connection(
                conn, task.id, role="assistant", stage="chat",
                content=(explanation + "\n\n" if index == 0 else "") + message.content,
                metadata={**message.metadata, "model_monitoring_proposal": proposal,
                          "monitoring_proposal_hash": proposal_hash},
            )

    shared_lane._driver(runtime).start(
        task_id=task.id, template_id="monitoring_run", tier=runtime.tier,
        slots={"experiment_id": contract.experiment_id, "dataset_id": contract.dataset_id,
               "target_col": contract.target_col,
               "monitoring_policy": {"thresholds": proposal["thresholds"]},
               "monitoring_binding": proposal["monitoring_binding"]},
        _persist_start_turn=persist,
    )
    return responses_lane.join_turn_response(repo, task.id)
