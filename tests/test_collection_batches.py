from concurrent.futures import ThreadPoolExecutor
import sqlite3

import pytest
from pydantic import ValidationError

from marvis.collection.batches import CollectionBatchRequest, CollectionBatchStore
from marvis.collection.ledger import CollectionEvidenceError
from marvis.db_schema import connect
from marvis.governance.repository import GovernanceRepository
from marvis.production_governance.repository import ProductionGovernanceRepository
from marvis.repositories.tasks import TaskRepository
from marvis.risk_context.source_contracts import SourceError
from tests.test_collection_cashflows import UNIT, ledger as ledger
from tests.test_collection_planning import AT, case, history, policy, spec


@pytest.fixture
def batch(ledger):
    cashflows, task, evidence = ledger
    settings = cashflows.settings
    governance = GovernanceRepository(settings.db_path)
    production = ProductionGovernanceRepository(settings.db_path)
    actors = {}
    for name, role in (
        ("maker", "maker"),
        ("other", "maker"),
        ("checker", "checker"),
        ("admin", "admin"),
    ):
        actor = governance.create_local_principal(display_name=name)
        production.claim_principal(
            local_principal_id=actor.id, display_name=name, role=role
        )
        actors[name] = actor.id
    p = policy(
        unit=UNIT,
        basis_artifact_id=evidence["source_artifact_id"],
        basis_artifact_hash=evidence["source_artifact_hash"],
    )
    request = CollectionBatchRequest(
        batch_id="batch-1",
        execution_mode="local_reference",
        strategy=spec(p).to_dict(),
        policy=p,
        cases=[case("case-1", subject_namespace="fixture", **evidence)],
        histories=[history(subject_namespace="fixture", **evidence)],
        as_of=AT,
        knowledge_cutoff=AT,
    )
    return CollectionBatchStore(settings), task, request, actors


def test_proposal_is_authenticated_idempotent_and_not_an_approval(batch):
    store, task, request, actors = batch
    result = store.prepare(task, request, actors["maker"])
    assert result["status"] == "proposed" and result["revision"] == 1
    assert result["execution_authorized"] is False
    assert result["preview"]["live_capacity_reserved"] is False
    assert result["preview_hash"] == result["preview"]["preview_hash"]
    assert result == store.prepare(task, request, actors["maker"])
    restarted = CollectionBatchStore(store.settings)
    assert result == restarted.read(task, request.batch_id, actors["checker"])
    assert result == restarted.read(task, request.batch_id, actors["admin"])
    with pytest.raises(CollectionEvidenceError, match="actor_forbidden"):
        restarted.read(task, request.batch_id, actors["other"])


def test_reference_only_no_implicit_real_customer_execution(batch):
    _, _, request, _ = batch
    with pytest.raises(ValidationError):
        CollectionBatchRequest.model_validate(
            {**request.model_dump(), "execution_mode": "production"}
        )


def test_scope_and_source_binding_are_checked_before_proposal_commit(batch):
    store, task, request, actors = batch
    payload = request.model_dump()
    payload["cases"][0]["subject_token"] = "f" * 64
    payload["histories"][0]["subject_token"] = "f" * 64
    with pytest.raises(CollectionEvidenceError, match="subject_mismatch"):
        store.prepare(
            task, CollectionBatchRequest.model_validate(payload), actors["maker"]
        )
    payload = request.model_dump()
    payload["cases"][0]["source_artifact_hash"] = "f" * 64
    with pytest.raises(CollectionEvidenceError, match="source_binding_invalid"):
        store.prepare(
            task, CollectionBatchRequest.model_validate(payload), actors["maker"]
        )
    with connect(store.settings.db_path) as conn:
        assert (
            conn.execute("SELECT count(*) FROM collection_batches").fetchone()[0] == 0
        )


def test_foreign_task_and_case_date_cannot_be_relabelled(batch):
    store, task, request, actors = batch
    with pytest.raises(CollectionEvidenceError, match="source_binding_invalid"):
        store.prepare("other-task", request, actors["maker"])
    payload = request.model_dump()
    payload["as_of"] = "2026-07-01T00:00:00Z"
    with pytest.raises(CollectionEvidenceError, match="precedes_case"):
        store.prepare(
            task, CollectionBatchRequest.model_validate(payload), actors["maker"]
        )


def test_cannot_change_money_unit_despite_equivalent_currency_label(batch):
    store, task, request, actors = batch
    payload = request.model_dump()
    payload["policy"]["unit"]["minor_unit_exponent"] = 0
    changed_policy = type(request.policy).model_validate(payload["policy"])
    payload["strategy"] = spec(changed_policy).to_dict()
    with pytest.raises(CollectionEvidenceError, match="currency_unit_mismatch"):
        store.prepare(
            task, CollectionBatchRequest.model_validate(payload), actors["maker"]
        )


def test_revoked_or_checker_principal_cannot_prepare(batch):
    store, task, request, actors = batch
    with pytest.raises(SourceError, match="role_forbidden"):
        store.prepare(task, request, actors["checker"])
    with connect(store.settings.db_path) as conn:
        conn.execute(
            "UPDATE production_principals SET status='revoked' WHERE local_principal_id=?",
            (actors["maker"],),
        )
    with pytest.raises(SourceError, match="role_forbidden"):
        store.prepare(task, request, actors["maker"])


def test_concurrent_duplicate_proposal_and_conflicting_actor(batch):
    store, task, request, actors = batch
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(lambda _: store.prepare(task, request, actors["maker"]), range(4))
        )
    assert all(result == results[0] for result in results)
    with pytest.raises(CollectionEvidenceError, match="idempotency_conflict"):
        store.prepare(task, request, actors["other"])
    payload = request.model_dump()
    payload["cases"][0]["features"]["dpd"] = 999
    with pytest.raises(CollectionEvidenceError, match="idempotency_conflict"):
        store.prepare(
            task, CollectionBatchRequest.model_validate(payload), actors["maker"]
        )


def test_immutable_identity_and_task_cleanup(batch):
    store, task, request, actors = batch
    store.prepare(task, request, actors["maker"])
    with connect(store.settings.db_path) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute("UPDATE collection_batches SET actor_id='changed'")
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute("DELETE FROM collection_batches")
    TaskRepository(store.settings.db_path).delete_task(task)
    with connect(store.settings.db_path) as conn:
        assert (
            conn.execute("SELECT count(*) FROM collection_batches").fetchone()[0] == 0
        )


def test_registered_source_file_drift_is_not_reauthenticated(batch):
    store, task, request, actors = batch
    (store.settings.tasks_dir / task / "source.json").write_text("replaced source")
    with pytest.raises(CollectionEvidenceError, match="source_integrity_failed"):
        store.prepare(task, request, actors["maker"])
