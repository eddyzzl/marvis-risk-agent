from marvis.decision_twin.batch import reconcile_batch, replay_batch
from marvis.decision_twin.batch_contracts import (
    HistoricalReconciliationRequest,
    HistoricalReplayRequest,
)
from marvis.decision_twin.batch_material import BatchMaterial
from marvis.settings import build_settings


def _result(receipt, task_id):
    payload = receipt["payload"]
    return {
        "schema_version": "decision_twin.tool_result.v2",
        "artifact_id": receipt["artifact_id"],
        "kind": receipt["kind"],
        "contract_hash": payload["contract_hash"],
        "authority": "proposal_only",
        "automatic_action_permitted": False,
        "evidence_url": f"/api/tasks/{task_id}/decision-twin/{receipt['artifact_id']}",
        "summary": {
            k: v
            for k, v in payload.items()
            if k not in {"scenarios", "records", "observed_actions", "contract"}
        },
        "scenarios": [
            {k: v for k, v in scenario.items() if k != "decisions"}
            for scenario in payload.get("scenarios", [])
        ],
    }


def tool_replay_history(inputs, ctx):
    material = BatchMaterial(build_settings(ctx.workspace), ctx.task_id)
    contract = HistoricalReplayRequest.model_validate(inputs["contract"])
    return _result(
        replay_batch(material, contract, inputs["proposal_hash"]), ctx.task_id
    )


def tool_reconcile_history(inputs, ctx):
    material = BatchMaterial(build_settings(ctx.workspace), ctx.task_id)
    contract = HistoricalReconciliationRequest.model_validate(inputs["contract"])
    return _result(
        reconcile_batch(material, contract, inputs["proposal_hash"]), ctx.task_id
    )
