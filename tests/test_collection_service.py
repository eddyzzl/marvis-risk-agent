"""Public collection material, ownership and feedback boundaries."""

import pytest

from marvis.collection.batches import CollectionBatchRequest
from marvis.collection.contracts import CollectionCase
from marvis.collection.ledger import CollectionEvidenceError
from marvis.collection.service import (
    CashflowImport,
    CollectionMaterial,
    CollectionService,
)
from marvis.db_schema import connect
from marvis.risk_context.source_contracts import SourceError
from tests.test_collection_batches import batch as batch
from tests.test_collection_cashflows import flow, ledger as ledger, request, schedule
from tests.test_collection_execution import executable
from tests.test_collection_planning import spec


@pytest.fixture
def public(batch):
    store, task, original, actors = batch
    svc = CollectionService(store.settings)
    value = CollectionMaterial(
        material_id="declared",
        description="Synthetic unverified historical cashflow and policy declarations",
        declarations={"synthetic": True},
    )
    evidence = svc.material(task, value, actors["maker"])
    refs = {k: evidence[k] for k in ("source_artifact_id", "source_artifact_hash")}
    with connect(store.settings.db_path) as conn:
        native = store.ledger._get(conn, task, "case", "case-1")
    native.update(case_id="public-case", **refs)
    case = CollectionCase.model_validate(native)
    svc.create_case(task, case, actors["maker"])
    return svc, task, actors, refs, case, original


def test_public_material_is_idempotent_scoped_and_immutable(public):
    svc, task, actors, refs, _, _ = public
    value = CollectionMaterial(
        material_id="declared",
        description="Synthetic unverified historical cashflow and policy declarations",
        declarations={"synthetic": True},
    )
    assert (
        svc.material(task, value, actors["maker"])["source_artifact_id"]
        == refs["source_artifact_id"]
    )
    with pytest.raises(CollectionEvidenceError, match="actor_forbidden"):
        svc.material(task, value, actors["other"])
    with pytest.raises(CollectionEvidenceError, match="identity_conflict"):
        svc.material(
            task, value.model_copy(update={"description": "changed"}), actors["maker"]
        )


def test_native_case_owner_cannot_be_claimed_by_other_maker(public):
    svc, task, actors, _, case, _ = public
    assert svc.case(task, case.case_id, actors["checker"])["case_id"] == case.case_id
    with pytest.raises(CollectionEvidenceError, match="actor_forbidden"):
        svc.case(task, case.case_id, actors["other"])
    with pytest.raises(CollectionEvidenceError, match="actor_forbidden"):
        svc.create_case(task, case, actors["other"])


def test_real_feedback_is_independent_of_estimates_and_retains_missing_coverage(public):
    svc, task, actors, refs, case, _ = public
    value = flow(refs, case_id=case.case_id)
    imported = svc.cashflows(task, CashflowImport(events=[value]), actors["maker"])
    assert imported == svc.cashflows(
        task, CashflowImport(events=[value]), actors["maker"]
    )
    incomplete = request(refs, case_id=case.case_id, coverage=[])
    unknown = svc.reconcile(task, incomplete, actors["maker"])
    assert unknown["amounts"] is None
    complete = request(refs, case_id=case.case_id)
    observed = svc.reconcile(task, complete, actors["maker"])
    assert observed["amounts"]["net_payments_minor"] == 3000
    assert observed["amounts"]["net_cost_minor"] == 0
    assert observed["source_truth_verified"] is False
    assert (
        svc.report(task, case.case_id, observed["receipt_hash"], actors["checker"])
        == observed
    )
    assert (
        svc.report(task, case.case_id, unknown["receipt_hash"], actors["maker"])
        == unknown
    )


@pytest.mark.parametrize("operation", ["schedule", "cashflows", "reconcile", "prepare"])
def test_every_public_writer_rejects_foreign_case_owner(public, operation):
    svc, task, actors, refs, case, original = public
    if operation == "schedule":
        data = schedule(refs).model_copy(update={"case_id": case.case_id})
    elif operation == "cashflows":
        data = CashflowImport(events=[flow(refs, case_id=case.case_id)])
    elif operation == "reconcile":
        data = request(refs, case_id=case.case_id)
    else:
        payload = executable(original).model_dump()
        payload["cases"][0].update(case_id=case.case_id, **refs)
        payload["histories"][0].update(**refs)
        payload["policy"].update(
            basis_artifact_id=refs["source_artifact_id"],
            basis_artifact_hash=refs["source_artifact_hash"],
        )
        from marvis.collection.actions import CollectionPolicy

        payload["strategy"] = spec(
            CollectionPolicy.model_validate(payload["policy"])
        ).to_dict()
        data = CollectionBatchRequest.model_validate(payload)
    with pytest.raises(CollectionEvidenceError, match="actor_forbidden"):
        getattr(svc, operation)(task, data, actors["other"])
    with connect(svc.settings.db_path) as conn:
        assert (
            conn.execute(
                "SELECT count(*) FROM collection_evidence WHERE kind IN ('flow','schedule','reconciliation')"
            ).fetchone()[0]
            == 0
        )
        assert (
            conn.execute("SELECT count(*) FROM collection_batches").fetchone()[0] == 0
        )


def test_role_revocation_between_precheck_and_writer_is_not_lost(public, monkeypatch):
    svc, task, actors, refs, case, _ = public
    original = svc.ledger.append

    def interleaving(*args, **kwargs):
        with connect(svc.settings.db_path) as conn:
            conn.execute(
                "UPDATE production_principals SET status='revoked' WHERE local_principal_id=?",
                (actors["maker"],),
            )
        return original(*args, **kwargs)

    monkeypatch.setattr(svc.ledger, "append", interleaving)
    with pytest.raises(SourceError):
        svc.cashflows(
            task,
            CashflowImport(events=[flow(refs, case_id=case.case_id)]),
            actors["maker"],
        )
    with connect(svc.settings.db_path) as conn:
        assert (
            conn.execute(
                "SELECT count(*) FROM collection_evidence WHERE kind='flow'"
            ).fetchone()[0]
            == 0
        )
