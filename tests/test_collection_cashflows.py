"""Real immutable ledger and independent minor-unit reconciliation examples."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import sqlite3

import pytest
from pydantic import ValidationError

from marvis.collection.contracts import (
    CashflowEvent,
    CollectionCase,
    InstallmentSchedule,
    ReconciliationRequest,
)
from marvis.collection.ledger import CollectionEvidenceError, CollectionLedger
from marvis.db_schema import connect, init_db
from marvis.repositories.task_artifacts import TaskArtifactRepository
from marvis.repositories.tasks import TaskRepository
from marvis.settings import build_settings
from tests.test_db import _task_create


UNIT = {
    "currency": "CNY",
    "minor_unit_exponent": 2,
    "definition_source": "synthetic fixed minor unit",
}


@pytest.fixture
def ledger(tmp_path):
    settings = build_settings(tmp_path / "workspace")
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    init_db(settings.db_path)
    settings.plugin_admin_token_path.parent.mkdir(parents=True, exist_ok=True)
    settings.plugin_admin_token_path.write_text("local-test-authentication-key")
    task = TaskRepository(settings.db_path).create_task(
        _task_create(task_type="strategy")
    )
    source = settings.tasks_dir / task.id / "source.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps({"fixture": "synthetic declared cashflow records"}))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    record = TaskArtifactRepository(settings.db_path).register(
        task_id=task.id,
        kind="collection_source",
        path=str(source.relative_to(settings.workspace)),
        content_hash=digest,
        origin_tool="synthetic_test",
        provenance={},
    )
    evidence = {"source_artifact_id": record["id"], "source_artifact_hash": digest}
    store = CollectionLedger(settings)
    case = CollectionCase(
        case_id="case-1",
        subject_namespace="fixture",
        subject_token="a" * 64,
        unit=UNIT,
        opening_balance_minor=10000,
        opened_at="2026-08-01T00:00:00Z",
        source_assurance="historical_import_unverified",
        **evidence,
    )
    store.create_case(task.id, case)
    return store, task.id, evidence


def flow(evidence, event_id="payment-1", **changes):
    return CashflowEvent.model_validate(
        {
            "source_id": "bank",
            "event_id": event_id,
            "case_id": "case-1",
            "kind": "payment",
            "amount_minor": 3000,
            "unit": UNIT,
            "event_at": "2026-08-10T00:00:00Z",
            "available_at": "2026-08-11T00:00:00Z",
            "source_assurance": "historical_import_unverified",
            **evidence,
            **changes,
        }
    )


def request(evidence, **changes):
    return ReconciliationRequest.model_validate(
        {
            "case_id": "case-1",
            "as_of": "2026-08-31T00:00:00Z",
            "knowledge_cutoff": "2026-09-01T00:00:00Z",
            "expected_sources": ["bank"],
            "coverage": [
                {
                    "source_id": "bank",
                    "from_at": "2026-08-01T00:00:00Z",
                    "through_at": "2026-08-31T00:00:00Z",
                    "available_at": "2026-09-01T00:00:00Z",
                    "status": "complete",
                    **evidence,
                }
            ],
            **changes,
        }
    )


def schedule(evidence):
    return InstallmentSchedule(
        schedule_id="plan-1",
        case_id="case-1",
        unit=UNIT,
        installments=[
            {
                "installment_id": "part-1",
                "due_at": "2026-08-15T00:00:00Z",
                "amount_minor": 5000,
            },
            {
                "installment_id": "part-2",
                "due_at": "2026-09-15T00:00:00Z",
                "amount_minor": 5000,
            },
        ],
        terms_artifact_id=evidence["source_artifact_id"],
        terms_artifact_hash=evidence["source_artifact_hash"],
    )


def test_report_is_listed_and_downloaded_through_shared_artifact_api(ledger):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from marvis.routers.artifacts import router

    store, task, evidence = ledger
    result = store.reconcile(task, request(evidence))
    app = FastAPI()
    app.state.settings = store.settings
    app.include_router(router)
    with TestClient(app) as client:
        listed = client.get(f"/api/tasks/{task}/task-artifacts").json()["artifacts"]
        report = next(
            row for row in listed if row["kind"] == "collection_cashflow_reconciliation"
        )
        assert report["available"] is True
        response = client.get(report["download_url"])
        assert response.status_code == 200
        assert hashlib.sha256(response.content).hexdigest() == result["receipt_hash"]
        assert response.json() == {
            k: v for k, v in result.items() if k != "receipt_hash"
        }


def test_task_deletion_cascades_but_live_case_evidence_cannot_be_deleted(ledger):
    store, task, evidence = ledger
    store.reconcile(task, request(evidence))
    with connect(store.settings.db_path) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute("DELETE FROM collection_evidence WHERE task_id=?", (task,))
    TaskRepository(store.settings.db_path).delete_task(task)
    with connect(store.settings.db_path) as conn:
        assert (
            conn.execute(
                "SELECT count(*) FROM collection_evidence WHERE task_id=?", (task,)
            ).fetchone()[0]
            == 0
        )


def test_gold_reconciliation_reversals_costs_and_installment_fulfillment(ledger):
    store, task, evidence = ledger
    store.register_schedule(task, schedule(evidence))
    part = {"schedule_id": "plan-1", "installment_id": "part-1"}
    rows = [
        flow(evidence, installment=part),
        flow(evidence, "payment-2", amount_minor=2000, installment=part),
        flow(
            evidence,
            "payment-undo",
            kind="payment_reversal",
            amount_minor=500,
            reverses={"source_id": "bank", "event_id": "payment-1"},
        ),
        flow(evidence, "cost-1", kind="cost", amount_minor=100),
        flow(
            evidence,
            "cost-undo",
            kind="cost_reversal",
            amount_minor=20,
            reverses={"source_id": "bank", "event_id": "cost-1"},
        ),
    ]
    store.append(task, list(reversed(rows)))  # importer order cannot change arithmetic
    report = store.reconcile(task, request(evidence, schedule_id="plan-1"))
    assert report["amounts"] == {
        "payment_minor": 5000,
        "payment_reversal_minor": 500,
        "cost_minor": 100,
        "cost_reversal_minor": 20,
        "net_payments_minor": 4500,
        "net_cost_minor": 80,
        "net_recovery_minor": 4420,
        "remaining_balance_minor": 5500,
        "overpayment_minor": 0,
    }
    assert [(p["status"], p["remaining_minor"]) for p in report["installments"]] == [
        ("overdue", 500),
        ("not_due", 5000),
    ]
    assert report["incremental_recovery_identified"] is False
    assert report["source_truth_verified"] is False
    assert report["external_action_executed"] is False


def test_real_restart_and_concurrent_exact_replays_do_not_double_count(ledger):
    store, task, evidence = ledger
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda _: CollectionLedger(store.settings).append(
                    task, [flow(evidence)]
                ),
                range(8),
            )
        )
    assert all(r == results[0] for r in results)
    restarted = CollectionLedger(store.settings)
    report = restarted.reconcile(task, request(evidence))
    assert report["stored_event_count"] == 1
    assert report["amounts"]["net_payments_minor"] == 3000
    assert restarted.read_report(task, report["receipt_hash"]) == report


def test_duplicate_conflict_rolls_back_the_entire_batch(ledger):
    store, task, evidence = ledger
    store.append(task, [flow(evidence)])
    with pytest.raises(CollectionEvidenceError, match="idempotency_conflict"):
        store.append(task, [flow(evidence, "other"), flow(evidence, amount_minor=1)])
    report = store.reconcile(task, request(evidence))
    assert report["stored_event_count"] == 1


@pytest.mark.parametrize(
    "changes, error",
    [
        ({"unit": {**UNIT, "currency": "USD"}}, "unit_mismatch"),
        ({"event_at": "2026-07-31T00:00:00Z"}, "case_or_unit_mismatch"),
        (
            {
                "kind": "payment_reversal",
                "reverses": {"source_id": "bank", "event_id": "missing"},
            },
            "original_missing",
        ),
    ],
)
def test_invalid_flow_bindings_are_rejected_without_partial_writes(
    ledger, changes, error
):
    store, task, evidence = ledger
    with pytest.raises(CollectionEvidenceError, match=error):
        store.append(task, [flow(evidence, **changes)])
    assert store.reconcile(task, request(evidence))["stored_event_count"] == 0


def test_partial_reversals_cannot_exceed_original_or_reverse_a_different_kind(ledger):
    store, task, evidence = ledger
    store.append(task, [flow(evidence)])
    ref = {"source_id": "bank", "event_id": "payment-1"}
    store.append(
        task,
        [
            flow(
                evidence,
                "undo-1",
                kind="payment_reversal",
                amount_minor=2000,
                reverses=ref,
            )
        ],
    )
    with pytest.raises(CollectionEvidenceError, match="exceeds_original"):
        store.append(
            task,
            [
                flow(
                    evidence,
                    "undo-2",
                    kind="payment_reversal",
                    amount_minor=1001,
                    reverses=ref,
                )
            ],
        )
    with pytest.raises(CollectionEvidenceError, match="reference_invalid"):
        store.append(
            task,
            [
                flow(
                    evidence,
                    "undo-3",
                    kind="cost_reversal",
                    amount_minor=1,
                    reverses=ref,
                )
            ],
        )
    assert (
        store.reconcile(task, request(evidence))["amounts"]["net_payments_minor"]
        == 1000
    )


def test_absent_coverage_and_unknown_availability_cannot_report_zero_recovery(ledger):
    store, task, evidence = ledger
    missing = store.reconcile(task, request(evidence, coverage=[]))
    assert missing["amounts"] is None
    assert missing["status"] == "insufficient_evidence"
    explicit = store.reconcile(task, request(evidence))
    assert explicit["amounts"]["net_recovery_minor"] == 0
    store.append(task, [flow(evidence, available_at=None)])
    unknown = store.reconcile(task, request(evidence))
    assert unknown["amounts"] is None
    assert "cashflow_availability_unknown" in unknown["reasons"]


def test_late_payment_does_not_rewrite_prior_receipt(ledger):
    store, task, evidence = ledger
    old = store.reconcile(task, request(evidence))
    store.append(task, [flow(evidence, available_at="2026-09-03T00:00:00Z")])
    earlier = store.reconcile(task, request(evidence))
    later = store.reconcile(
        task, request(evidence, knowledge_cutoff="2026-09-04T00:00:00Z")
    )
    assert earlier["amounts"]["net_payments_minor"] == 0
    assert later["amounts"]["net_payments_minor"] == 3000
    assert store.read_report(task, old["receipt_hash"]) == old


def test_maturity_requires_explicit_policy_and_complete_observation_window(ledger):
    store, task, evidence = ledger
    assert (
        store.reconcile(task, request(evidence))["maturity"]["status"] == "not_declared"
    )
    policy = {
        "anchor": "case_opened_at",
        "observation_days": 31,
        "policy_artifact_id": evidence["source_artifact_id"],
        "policy_artifact_hash": evidence["source_artifact_hash"],
    }
    assert (
        store.reconcile(task, request(evidence, maturity=policy))["maturity"]["status"]
        == "immature"
    )
    policy["observation_days"] = 30
    mature = store.reconcile(task, request(evidence, maturity=policy))
    assert mature["maturity"]["status"] == "mature_under_declared_policy"
    assert mature["maturity"]["mature_outcome_verified"] is True
    incomplete = store.reconcile(task, request(evidence, maturity=policy, coverage=[]))
    assert incomplete["maturity"]["mature_outcome_verified"] is False


def test_reversal_with_invisible_original_is_unknown(ledger):
    store, task, evidence = ledger
    store.append(
        task,
        [
            flow(evidence, available_at="2026-09-03T00:00:00Z"),
            flow(
                evidence,
                "undo",
                kind="payment_reversal",
                amount_minor=1000,
                reverses={"source_id": "bank", "event_id": "payment-1"},
            ),
        ],
    )
    report = store.reconcile(task, request(evidence))
    assert report["amounts"] is None
    assert "reversal_original_unavailable_at_cutoff" in report["reasons"]


def test_unexpected_sources_and_incomplete_watermarks_remain_unknown(ledger):
    store, task, evidence = ledger
    store.append(task, [flow(evidence, source_id="unlisted")])
    report = store.reconcile(task, request(evidence))
    assert report["amounts"] is None
    assert "undeclared_event_source:unlisted" in report["reasons"]
    req = request(evidence).model_dump()
    req["coverage"][0]["status"] = "partial"
    report = store.reconcile(task, ReconciliationRequest.model_validate(req))
    assert "incomplete_source_coverage:bank" in report["reasons"]


def test_unallocated_payments_do_not_fabricate_installment_fulfillment(ledger):
    store, task, evidence = ledger
    store.register_schedule(task, schedule(evidence))
    store.append(task, [flow(evidence, amount_minor=12000)])
    report = store.reconcile(task, request(evidence, schedule_id="plan-1"))
    assert report["amounts"]["remaining_balance_minor"] == 0
    assert report["amounts"]["overpayment_minor"] == 2000
    assert report["installments"][0]["status"] == "overdue"
    assert report["unallocated_payment_count"] == 1


def test_source_tamper_and_cross_task_binding_rejected(ledger):
    store, task, evidence = ledger
    second = TaskRepository(store.settings.db_path).create_task(
        _task_create(task_type="strategy")
    )
    case = CollectionCase(
        case_id="case-2",
        subject_namespace="fixture",
        subject_token="b" * 64,
        unit=UNIT,
        opening_balance_minor=100,
        opened_at="2026-08-01T00:00:00Z",
        source_assurance="historical_import_unverified",
        **evidence,
    )
    with pytest.raises(CollectionEvidenceError, match="source_binding_invalid"):
        store.create_case(second.id, case)
    (store.settings.tasks_dir / task / "source.json").write_text("changed")
    with pytest.raises(CollectionEvidenceError, match="source_integrity_failed"):
        store.append(task, [flow(evidence)])


def test_immutable_database_rows_and_hmac_detect_tampering(ledger):
    store, task, evidence = ledger
    store.append(task, [flow(evidence)])
    with connect(store.settings.db_path) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute("UPDATE collection_evidence SET case_id='other'")
        conn.execute("DROP TRIGGER collection_evidence_no_update")
        conn.execute(
            "UPDATE collection_evidence SET signature=? WHERE kind='flow'", ("0" * 64,)
        )
    with pytest.raises(CollectionEvidenceError, match="integrity_failed"):
        store.reconcile(task, request(evidence))


def test_report_and_registry_commit_or_rollback_together(ledger, monkeypatch):
    store, task, evidence = ledger

    def fail(*args, **kwargs):
        raise RuntimeError("injected registration failure")

    with monkeypatch.context() as patch:
        patch.setattr(TaskArtifactRepository, "register_on_connection", fail)
        with pytest.raises(RuntimeError, match="injected"):
            store.reconcile(task, request(evidence))
    with connect(store.settings.db_path) as conn:
        assert (
            conn.execute(
                "SELECT count(*) FROM collection_evidence WHERE kind='reconciliation'"
            ).fetchone()[0]
            == 0
        )
    assert not list((store.settings.tasks_dir / task / "collection").glob("*.json"))
    report = store.reconcile(task, request(evidence))
    rows = TaskArtifactRepository(store.settings.db_path).list_for_task(task)
    assert (
        len([r for r in rows if r["kind"] == "collection_cashflow_reconciliation"]) == 1
    )
    assert store.reconcile(task, request(evidence)) == report
    path = (
        store.settings.tasks_dir
        / task
        / "collection"
        / f"reconciliation-{report['receipt_hash']}.json"
    )
    path.write_text("tampered")
    with pytest.raises(CollectionEvidenceError, match="integrity_failed"):
        store.read_report(task, report["receipt_hash"])


@pytest.mark.parametrize(
    "changes",
    [
        {"amount_minor": 1.5},
        {"amount_minor": True},
        {"amount_minor": 0},
        {"event_at": "2026-08-10T00:00:00"},
        {"available_at": "2026-08-09T00:00:00Z"},
    ],
)
def test_money_and_time_contract_rejects_ambiguous_values(changes):
    with pytest.raises(ValidationError):
        flow(
            {"source_artifact_id": "source", "source_artifact_hash": "a" * 64},
            **changes,
        )
