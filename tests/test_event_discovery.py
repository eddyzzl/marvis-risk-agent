from marvis.db_schema import connect
from test_event_runtime import runtime as runtime, prepare


def test_discovery_preserves_principal_scope_revocation_and_authenticated_originals(runtime):
    rt = runtime
    proposal = prepare(rt)
    base = f"/api/tasks/{rt.task.id}/risk-events"
    mine = rt.maker.get(base)
    assert mine.status_code == 200
    payload = mine.json()
    assert payload["sources"][0]["contract_hash"] == rt.source.contract_hash
    assert payload["grants"][0]["owned"] and payload["grants"][0]["state"] == "active"
    assert payload["requests"][0]["proposal_hash"] == proposal["proposal_hash"]
    assert "claims" not in mine.text and "signature" not in mine.text
    assert rt.other.get(base).json() == {"sources": [], "grants": [], "requests": []}
    admin = rt.admin.get(base).json()
    assert admin["sources"] and not admin["grants"][0]["owned"] and admin["requests"] == []
    revoked = rt.admin.post(base + f"/grants/{rt.grant['grant_id']}/revoke")
    assert revoked.status_code == 200
    after = rt.maker.get(base).json()
    assert after["grants"][0]["state"] == "revoked" and after["requests"] == []
    # This private disposable DB deliberately defeats its UPDATE guard to
    # exercise the reader's separate original-MAC validation, not a live write.
    with connect(rt.settings.db_path) as conn:
        conn.execute("DROP TRIGGER event_grants_immutable")
        conn.execute("UPDATE event_grants SET signature=? WHERE id=?", ("0" * 64, rt.grant["grant_id"]))
    response = rt.maker.get(base)
    assert response.status_code == 409 and response.json()["detail"]["code"] == "event_evidence_integrity_failed"
