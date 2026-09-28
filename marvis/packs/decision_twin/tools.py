from marvis.governance.native_producers import NativeInvocation
from marvis.decision_twin.producer_output import tool_result as _result
from marvis.decision_twin.batch import reconcile_batch, replay_batch
from marvis.decision_twin.batch_contracts import (
    HistoricalReconciliationRequest,
    HistoricalReplayRequest,
)
from marvis.decision_twin.batch_material import BatchMaterial
from marvis.settings import build_settings


def tool_replay_history(inputs, ctx):
    material = BatchMaterial(
        build_settings(ctx.workspace),
        ctx.task_id,
        invocation=NativeInvocation.from_context(
            ctx, "decision_twin.replay_history", inputs
        ),
    )
    contract = HistoricalReplayRequest.model_validate(inputs["contract"])
    return _result(
        replay_batch(material, contract, inputs["proposal_hash"]), ctx.task_id
    )


def tool_reconcile_history(inputs, ctx):
    material = BatchMaterial(
        build_settings(ctx.workspace),
        ctx.task_id,
        invocation=NativeInvocation.from_context(
            ctx, "decision_twin.reconcile_history", inputs
        ),
    )
    contract = HistoricalReconciliationRequest.model_validate(inputs["contract"])
    return _result(
        reconcile_batch(material, contract, inputs["proposal_hash"]), ctx.task_id
    )
