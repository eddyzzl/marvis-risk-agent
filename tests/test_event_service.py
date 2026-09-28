# ruff: noqa: F811
# Imported pytest fixture is intentionally requested by parameter name.
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from marvis.db_schema import connect
from marvis.risk_context.event_contracts import EventError, content_hash
from marvis.risk_context.event_service import (
    EventRequest,
    EventService,
    EventToolInput,
    KIND,
)
from marvis.risk_context.source_contracts import SourceError
from test_event_features import T, event, coverage, query
from test_event_repository import runtime, ingest  # noqa: F401


def prepare(rt, request_id="event-one", **changes):
    service = EventService(rt.settings)
    payload = EventRequest(
        request_id=request_id,
        grant_id=rt.grant.grant_id,
        contract=query(rt.source, **changes),
    )
    proposal = service.prepare(rt.task.id, payload, rt.actors["maker"])
    inputs = EventToolInput(
        request_id=request_id,
        proposal_hash=proposal["proposal_hash"],
        contract=payload.contract,
    )
    rt.clock["at"] = T
    return service, payload, proposal, inputs


def evidence(rt, service, request_id="event-one"):
    return service.evidence(
        rt.task.id, request_id, rt.actors["maker"], rt.grant.grant_id
    )


def test_registered_native_evidence_survives_restart_and_exact_duplicate(runtime):
    rt = runtime
    ingest(rt, [event()])
    ingest(rt, [coverage()], kind="coverage")
    service, payload, proposal, inputs = prepare(rt)
    assert service.prepare(rt.task.id, payload, rt.actors["maker"]) == proposal
    result = service.execute(rt.task.id, inputs)
    assert result["status"] == "measured"
    assert result["features"]["amount"]["value"] == 100
    stored = evidence(rt, service)
    assert stored["artifact_id"] == result["artifact_id"]
    assert content_hash(stored["receipt"]["snapshot"]) == result["snapshot_hash"]
    restarted = EventService(rt.settings)
    assert evidence(rt, restarted) == stored
    assert restarted.execute(rt.task.id, inputs) == result
    assert (
        len(
            [
                a
                for a in service.artifacts.list_for_task(rt.task.id)
                if a["kind"] == KIND
            ]
        )
        == 1
    )


def test_concurrent_publication_same_request_is_idempotent(runtime):
    rt = runtime
    service, _, _, inputs = prepare(rt)
    with ThreadPoolExecutor(4) as executor:
        results = list(
            executor.map(
                lambda _: EventService(rt.settings).execute(rt.task.id, inputs),
                range(4),
            )
        )
    assert all(result == results[0] for result in results)
    assert evidence(rt, service)["artifact_id"] == results[0]["artifact_id"]


@pytest.mark.parametrize("change", ["hash", "contract", "unknown_request"])
def test_reviewed_intent_cannot_be_replaced(runtime, change):
    rt = runtime
    service, _, _, inputs = prepare(rt)
    updates = {
        "hash": {"proposal_hash": "a" * 64},
        "contract": {"contract": query(rt.source, window_seconds=120)},
        "unknown_request": {"request_id": "not-issued"},
    }[change]
    with pytest.raises(
        EventError, match="event_reviewed_contract_mismatch|event_request_not_found"
    ):
        service.execute(rt.task.id, inputs.model_copy(update=updates))
    with connect(rt.settings.db_path) as conn:
        assert conn.execute("SELECT count(*) FROM event_decisions").fetchone()[0] == 0


def test_task_grantee_grant_and_input_identity_are_bound(runtime):
    rt = runtime
    service, payload, _, inputs = prepare(rt)
    with pytest.raises(EventError, match="event_request_not_found"):
        service.execute(rt.other_task.id, inputs)
    with pytest.raises(EventError, match="event_grant_scope_forbidden"):
        service.prepare(rt.task.id, payload, rt.actors["other"])
    with pytest.raises(EventError, match="event_request_id_conflict"):
        service.prepare(
            rt.task.id,
            payload.model_copy(
                update={"contract": query(rt.source, window_seconds=90)}
            ),
            rt.actors["maker"],
        )
    rt.repo.revoke_grant(rt.task.id, rt.grant.grant_id, rt.actors["admin"])
    with pytest.raises(EventError, match="event_grant_not_active"):
        service.execute(rt.task.id, inputs)


@pytest.mark.parametrize(
    "tamper", ["file", "origin", "provenance", "intent", "symlink"]
)
def test_all_evidence_carriers_are_verified(runtime, tamper):
    rt = runtime
    service, _, _, inputs = prepare(rt)
    result = service.execute(rt.task.id, inputs)
    artifact = service.artifacts.get_for_task(rt.task.id, result["artifact_id"])
    path = Path(artifact["path"])
    if tamper == "file":
        path.write_text("{}")
    elif tamper == "symlink":
        replacement = rt.path / "copied.json"
        replacement.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(replacement)
    else:
        with connect(rt.settings.db_path) as conn:
            conn.execute("DROP TRIGGER trg_task_artifacts_immutable_update")
            if tamper == "intent":
                conn.execute("DROP TRIGGER event_requests_immutable")
                conn.execute(
                    "UPDATE event_requests SET payload=json_set(payload,'$.actor_id','forged')"
                )
            elif tamper == "origin":
                conn.execute(
                    "UPDATE task_artifacts SET origin_tool='forged' WHERE id=?",
                    (artifact["id"],),
                )
            else:
                conn.execute(
                    "UPDATE task_artifacts SET provenance_json='{}' WHERE id=?",
                    (artifact["id"],),
                )
    with pytest.raises(EventError, match="integrity_failed|binding_failed"):
        evidence(rt, service)
    if tamper in {"file", "symlink", "intent"}:
        with pytest.raises(EventError, match="integrity_failed"):
            service.execute(rt.task.id, inputs)


def test_file_publish_failure_recovery_keeps_the_original_receipt(runtime, monkeypatch):
    rt = runtime
    service, _, _, inputs = prepare(rt)
    native_register = service.artifacts.register_on_connection

    def fail(*args, **kwargs):
        raise RuntimeError("test publication failure")

    monkeypatch.setattr(service.artifacts, "register_on_connection", fail)
    with pytest.raises(RuntimeError, match="publication failure"):
        service.execute(rt.task.id, inputs)
    with connect(rt.settings.db_path) as conn:
        original = conn.execute("SELECT payload FROM event_decisions").fetchone()[0]
    monkeypatch.setattr(service.artifacts, "register_on_connection", native_register)
    result = service.execute(rt.task.id, inputs)
    assert result["artifact_id"]
    with connect(rt.settings.db_path) as conn:
        assert (
            conn.execute("SELECT payload FROM event_decisions").fetchone()[0]
            == original
        )


def test_active_original_actor_is_required_in_worker(runtime):
    rt = runtime
    service, _, _, inputs = prepare(rt)
    with connect(rt.settings.db_path) as conn:
        conn.execute(
            "UPDATE local_principals SET expires_at='2020-01-01T00:00:00+00:00' WHERE id=?", (rt.actors["maker"],)
        )
    with pytest.raises(SourceError, match="source_role_forbidden"):
        service.execute(rt.task.id, inputs)


def test_artifact_parent_symlink_cannot_escape_task(runtime):
    rt = runtime
    service, _, _, inputs = prepare(rt)
    outside = rt.path / "outside"
    outside.mkdir()
    (rt.settings.tasks_dir / rt.task.id / "event_evidence").symlink_to(outside)
    with pytest.raises(EventError, match="event_artifact_path_invalid"):
        service.execute(rt.task.id, inputs)
    assert list(outside.iterdir()) == []
