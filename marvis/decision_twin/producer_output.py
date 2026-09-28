"""Exact native Tool output shared by publication and the normal return path."""


def tool_result(receipt, task_id):
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
