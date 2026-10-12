"""Synthetic local checks from real product HTTP/ToolRunner carriers.

These cases are deliberately development inputs, not a hidden acceptance suite.
Reference calculations never call the producer's evaluators or ledger methods.
"""
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
from types import SimpleNamespace
import sqlite3

from fastapi.testclient import TestClient
import pandas as pd
import pytest
from pydantic import TypeAdapter, ValidationError

from marvis.app import create_app
from marvis.db import StrategyRepository
from marvis.db_schema import connect
from marvis.domain import TaskCreate
from marvis.orchestrator.eval.runtime_archive_collections import FrozenCollectionsBinding, revalidate_collections_archive
from marvis.orchestrator.eval.runtime_archive_connector import FrozenConnectorBinding, revalidate_connector_archive
from marvis.orchestrator.eval.runtime_archive_online import FrozenOnlineBinding, revalidate_online_archive
from marvis.orchestrator.eval.runtime_collections_reference import batch_preview_reference
from marvis.orchestrator.eval.runtime_connector_reference import event_features_reference, source_assessment_reference
from marvis.orchestrator.eval.runtime_contracts import digest
from marvis.orchestrator.eval.runtime_custody import capture_case, prepare_custody_run
from marvis.orchestrator.eval.runtime_extensions_contracts import ExtensionBusinessCase
from marvis.orchestrator.eval.runtime_extension_inputs import ExtensionBusinessInput
from marvis.orchestrator.eval.runtime_extensions_archive import measured_values
from marvis.packs.strategy.dsl import StrategyAction, StrategyRuleSpec, StrategySpec
from marvis.packs.strategy.strategy import build_strategy_from_spec
from marvis.repositories.tasks import TaskRepository

from test_collection_cashflows import UNIT, flow, request, schedule
from test_collection_planning import case as allocation_case, history, policy, spec
from test_event_features import coverage, event, query, source
from test_event_runtime import gated_plan, approve
from test_operations_api import _claim_role
from test_reference_deployment import _install_activate, _promotion


def native(tmp_path, **app_options):
    app = create_app(tmp_path / "workspace", **app_options)
    settings = app.state.settings
    task = TaskRepository(settings.db_path).create_task(TaskCreate(model_name="Extension synthetic local check", model_version="test", validator="qa", source_dir=str(tmp_path), task_type="strategy", run_mode="manual"))
    maker, checker, admin = [TestClient(app) for _ in range(3)]
    actors = {role: _claim_role(app, client, role)["id"] for role, client in zip(("maker", "checker", "admin"), (maker, checker, admin), strict=True)}
    materials = settings.tasks_dir / task.id / "materials"
    materials.mkdir(parents=True)
    return SimpleNamespace(app=app, settings=settings, task=task, maker=maker, checker=checker, admin=admin, actors=actors, materials=materials, root=tmp_path)


def rows(rt, sql, parameters=()):
    with connect(rt.settings.db_path) as conn:
        return [dict(row) for row in conn.execute(sql, parameters).fetchall()]


def case_for(rt, family, actions, materials=()):
    return ExtensionBusinessCase.model_validate({"id": family, "revision": "synthetic-local-1", "family": family, "task": {"task_type": "strategy"}, "actions": actions, "materials": list(materials), "business_constraints_source": "synthetic local product checks; no original hidden acceptance answers"})


def common(rt, case):
    return {"case": case, "run_id": "synthetic-local-run", "task_id": rt.task.id, "cases_sha256": digest(case.model_dump()), "expected_sha256": digest({"purpose": "synthetic local check, not acceptance expected"}), "source": {"commit": "0" * 40, "source_sha256": "b" * 64, "dirty_diff_sha256": "c" * 64}, "model_connection_sha256": "d" * 64}


def archive(rt, binding):
    run = prepare_custody_run(rt.root / "private", binding.run_id, forbidden=(rt.materials,))
    metadata = {key: getattr(binding, key) for key in ("run_id", "cases_sha256", "expected_sha256", "source", "model_connection_sha256")}
    metadata["execution_before_scoring"] = {"execution": {"task_id": rt.task.id}}
    receipt = capture_case(run_dir=run, workspace=rt.settings.workspace, dataset_root=rt.materials, case=binding.case, private={"scope": "synthetic product carrier; no expected answers"}, bindings=metadata, owned_processes_stopped=True)
    return run / binding.case.id, receipt["manifest_sha256"]


def repin_synthetic_database(path):
    """A new caller-held byte pin must not bless fabricated measured values."""
    manifest_path = path / "manifest.json"
    value = json.loads(manifest_path.read_bytes())
    database = path / "workspace/marvis.sqlite"
    value["files"]["workspace/marvis.sqlite"]["sha256"] = digest(database.read_bytes())
    value["files"]["workspace/marvis.sqlite"]["size_bytes"] = database.stat().st_size
    manifest_path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return digest(manifest_path.read_bytes())


def require(response, status):
    assert response.status_code == status, response.text
    return response.json()


def show_synthetic_reference_failure(monkeypatch):
    from marvis.orchestrator.eval.runtime_extensions_archive import ExtensionArchive
    original = ExtensionArchive.__exit__
    def leave(self, kind, exc, traceback):
        result = original(self, kind, exc, traceback)
        if exc is not None:
            raise AssertionError(f"synthetic independent reference error: {type(exc).__name__}: {exc}")
        return result
    monkeypatch.setattr(ExtensionArchive, "__exit__", leave)


@pytest.fixture
def collection_originals(tmp_path):
    rt = native(tmp_path)
    prefix = f"/api/tasks/{rt.task.id}/collection"
    evidence = require(rt.maker.post(prefix + "/materials", json={"material_id": "declared", "description": "Synthetic cashflow declarations", "declarations": {"synthetic": True}}), 201)
    refs = {key: evidence[key] for key in ("source_artifact_id", "source_artifact_hash")}
    declared_case = {"case_id": "case-1", "subject_namespace": "fixture", "subject_token": "a" * 64, "unit": UNIT, "opening_balance_minor": 10000, "opened_at": "2026-08-01T00:00:00Z", "source_assurance": "historical_import_unverified", **refs}
    actual_case = require(rt.maker.post(prefix + "/cases", json=declared_case), 201)
    sch = schedule(refs).model_dump()
    require(rt.maker.post(prefix + "/schedules", json=sch), 201)
    events = [flow(refs, installment={"schedule_id": "plan-1", "installment_id": "part-1"}), flow(refs, "reverse", kind="payment_reversal", amount_minor=1000, reverses={"source_id": "bank", "event_id": "payment-1"}), flow(refs, "late", amount_minor=800, available_at="2026-09-10T00:00:00Z"), flow(refs, "cost", kind="cost", amount_minor=200)]
    payload = {"events": [row.model_dump() for row in events]}
    require(rt.maker.post(prefix + "/cashflows", json=payload), 201)
    req = request(refs, schedule_id="plan-1")
    report = require(rt.maker.post(prefix + "/reconciliations", json=req.model_dump()), 201)
    ledger = rows(rt, "SELECT * FROM collection_evidence WHERE task_id=? AND case_id=? ORDER BY kind,identity", (rt.task.id, "case-1"))
    artifact = rows(rt, "SELECT * FROM task_artifacts WHERE task_id=? AND kind='collection_cashflow_reconciliation'", (rt.task.id,))[0]
    actions = [{"kind": "create_collection_case", "actor": "maker", "request": actual_case}, {"kind": "create_collection_schedule", "actor": "maker", "request": sch}, {"kind": "import_collection_cashflows", "actor": "maker", "request": payload}, {"kind": "reconcile_collection_cashflows", "actor": "maker", "request": req.model_dump()}]
    case = case_for(rt, "collections", actions)
    binding = FrozenCollectionsBinding(**common(rt, case), reconciliations=[{"request": req, "receipt_hash": report["receipt_hash"], "artifact_id": artifact["id"], "ledger_records_sha256": digest(ledger)}])
    path, sha = archive(rt, binding)
    return rt, path, sha, binding


def test_real_http_collection_archive_recomputes_reversals_late_visibility_and_installments(collection_originals, monkeypatch):
    _, path, sha, binding = collection_originals
    monkeypatch.setattr("marvis.collection.ledger.CollectionLedger.reconcile", lambda *a, **k: pytest.fail("producer used as proof"))
    value = revalidate_collections_archive(path, expected_manifest_sha256=sha, frozen_binding=binding)
    assert value["supported_checks_verified"], value
    report = value["values"]["reconciliations"][0]
    assert report["amounts"]["net_payments_minor"] == 2000
    assert report["amounts"]["net_recovery_minor"] == 1800
    assert report["visible_event_count"] == 3
    assert value["source_authentication"] == value["acceptance_claim"] == "not_established"
    assert not (path / "workspace/plugin_admin_token").exists()


def test_wrong_manifest_or_external_ledger_pin_does_not_recompute_as_pass(collection_originals):
    _, path, sha, binding = collection_originals
    wrong = revalidate_collections_archive(path, expected_manifest_sha256="f" * 64, frozen_binding=binding)
    assert not wrong["supported_checks_verified"]
    item = binding.reconciliations[0].model_copy(update={"ledger_records_sha256": "e" * 64})
    result = revalidate_collections_archive(path, expected_manifest_sha256=sha, frozen_binding=binding.model_copy(update={"reconciliations": [item]}))
    assert not result["supported_checks_verified"]


def test_rehashed_fabricated_collection_amount_is_rejected_by_independent_math(collection_originals):
    _, path, _, binding = collection_originals
    with sqlite3.connect(path / "workspace/marvis.sqlite") as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM collection_evidence WHERE kind='reconciliation'").fetchone()
        value = json.loads(row["body_json"])
        value["amounts"]["net_payments_minor"] += 777
        conn.execute("DROP TRIGGER collection_evidence_no_update")
        conn.execute("UPDATE collection_evidence SET body_json=?,body_hash=? WHERE kind='reconciliation'", (json.dumps(value), digest(value)))
        inventory = [dict(row) for row in conn.execute("SELECT * FROM collection_evidence WHERE task_id=? AND case_id=? ORDER BY kind,identity", (binding.task_id, "case-1"))]
    sha = repin_synthetic_database(path)
    item = binding.reconciliations[0].model_copy(update={"ledger_records_sha256": digest(inventory)})
    result = revalidate_collections_archive(path, expected_manifest_sha256=sha, frozen_binding=binding.model_copy(update={"reconciliations": [item]}))
    assert result["checks"]["archive_integrity"]["status"] == "verified"
    assert result["checks"]["independent_reconciliation"]["status"] == "mismatch"
    assert not result["supported_checks_verified"]


def test_collection_preview_independent_money_frequency_and_queue_order():
    p = policy()
    body = {"policy": p.model_dump(), "strategy": spec(p).to_dict(), "cases": [allocation_case("case-2").model_dump(), allocation_case("case-1").model_dump()], "histories": [history().model_dump()], "as_of": "2026-08-10T10:00:00Z", "knowledge_cutoff": "2026-08-10T10:00:00Z"}
    from marvis.collection.planning import plan_collection_actions
    from marvis.collection.actions import CollectionCaseInput, ContactHistory
    actual = plan_collection_actions(body["strategy"], p, [CollectionCaseInput.model_validate(row) for row in body["cases"]], [ContactHistory.model_validate(row) for row in body["histories"]], as_of=body["as_of"], knowledge_cutoff=body["knowledge_cutoff"])
    assert batch_preview_reference(body) == actual


@pytest.fixture
def online_originals(tmp_path):
    from marvis.operations import MonitoringOutcome
    fixed = datetime(2026, 10, 1, 10, 0, tzinfo=UTC)
    rt = native(tmp_path, operations_clock=lambda: fixed, operations_executor_allowlist={"synthetic-local": lambda req: MonitoringOutcome(level="green", evidence_ref="synthetic-period", evidence_hash=digest({"period": req.period.key}))})
    declaration = StrategySpec(strategy_type="approval", rules=(StrategyRuleSpec(rule_id="low", priority=1, condition={"op": "compare", "field": "x1", "operator": "<", "value": .5}, action=StrategyAction(type="approval", reason_code="LOW_RULE")),), default_action=StrategyAction(type="review", reason_code="RULE_REVIEW"))
    strategy = replace(build_strategy_from_spec(declaration, score_col="x1", description="Synthetic rules"), id="online-strategy")
    StrategyRepository(rt.settings.db_path).create_strategy(rt.task.id, strategy)
    with connect(rt.settings.db_path) as conn:
        conn.execute("UPDATE strategies SET asset_status='validated' WHERE id='online-strategy'")
    package_request = {"package_kind": "rule_only", "strategy_id": strategy.id, "strategy_version": 1, "decision_node": "underwriting", "raw_schema": [{"name": "x1", "type": "number"}, {"name": "x2", "type": "number"}]}
    package = require(rt.maker.post("/api/reference-decision/packages", json=package_request), 201)
    governed = (rt.app, rt.maker, rt.checker, rt.admin)
    promotion = _promotion(governed, package["package_hash"])
    _, deployment = _install_activate(governed, promotion)
    decisions, actions = [], [{"kind": "build_decision_package", "actor": "maker", "request": package_request}]
    for index, feature in enumerate((.1, .9)):
        req = {"request_id": f"req-{index}", "decision_node": "underwriting", "expected_package_hash": package["package_hash"], "features": {"x1": feature, "x2": 2}}
        first = require(rt.maker.post("/api/reference-decision/decisions", json=req), 200)
        assert require(rt.maker.post("/api/reference-decision/decisions", json=req), 200) == first
        actions.append({"kind": "submit_reference_decision", "actor": "maker", "request": req})
        record = rows(rt, "SELECT * FROM reference_decisions WHERE environment=? AND request_id=?", ("local-reference:production", req["request_id"]))[0]
        decisions.append({"request": req, "slot": "production", "record_sha256": digest(record)})
    records = {"deployment": rows(rt, "SELECT * FROM production_deployments WHERE id=?", (deployment["id"],))[0], "promotion": rows(rt, "SELECT * FROM production_promotion_requests WHERE id=?", (promotion["id"],))[0], "approvals": rows(rt, "SELECT * FROM production_promotion_approvals WHERE request_id=? ORDER BY stage", (promotion["id"],)), "installation": rows(rt, "SELECT * FROM reference_installations WHERE promotion_id=?", (promotion["id"],))[0], "manifest": rows(rt, "SELECT * FROM production_deployment_manifests WHERE id=?", (deployment["manifest_hash"],))[0], "head": rows(rt, "SELECT * FROM production_environment_heads WHERE environment='local-reference'")[0]}
    schedule_payload = {"expected_revision": 0, "schedule": {"schedule_id": "synthetic-schedule", "revision": 1, "monitoring_ref": "synthetic-local", "active_from": (fixed - timedelta(minutes=2)).isoformat(), "calendar": {"anchor_at": (fixed - timedelta(minutes=2)).isoformat(), "interval_seconds": 60}, "catch_up_budget": 1, "lease_seconds": 30, "retry_policy": {"max_attempts": 2, "initial_backoff_seconds": 0., "multiplier": 1., "max_backoff_seconds": 0.}, "enabled": True}}
    require(rt.maker.post("/api/operations/schedules", json=schedule_payload), 201)
    require(rt.admin.post("/api/operations/tick", json={"catch_up_budget": 1, "notification_budget": 1}), 200)
    actions.append({"kind": "publish_operations_schedule", "actor": "maker", "request": schedule_payload})
    schedule_records = {table: sorted(rows(rt, f"SELECT * FROM {table} WHERE schedule_id=?", ("synthetic-schedule",)), key=digest) for table in ("operations_periods", "operations_runs", "operations_outcomes", "operations_notification_outbox")}
    schedule_records["schedule"] = rows(rt, "SELECT * FROM operations_schedules WHERE schedule_id=? AND revision=1", ("synthetic-schedule",))[0]
    case = case_for(rt, "online_decision", actions)
    binding = FrozenOnlineBinding(**common(rt, case), package_hash=package["package_hash"], package_record_sha256=digest(rows(rt, "SELECT * FROM reference_decision_packages WHERE id=?", (package["package_hash"],))[0]), deployment_id=deployment["id"], deployment_records_sha256=digest(records), production_audit_head_sha256=rows(rt, "SELECT event_hash FROM production_governance_events ORDER BY sequence DESC LIMIT 1")[0]["event_hash"], decisions=decisions, schedules=[{"schedule_id": "synthetic-schedule", "revision": 1, "records_sha256": digest(schedule_records)}])
    path, sha = archive(rt, binding)
    return rt, path, sha, binding


def test_real_worker_online_archive_independently_checks_first_match_and_role_chain(online_originals, monkeypatch):
    _, path, sha, binding = online_originals
    monkeypatch.setattr("marvis.reference_decision.evaluation.evaluate", lambda *a, **k: pytest.fail("producer used as proof"))
    result = revalidate_online_archive(path, expected_manifest_sha256=sha, frozen_binding=binding)
    assert result["supported_checks_verified"], result["checks"]
    assert [row["action_type"] for row in result["values"]["decisions"]] == ["approval", "review"]
    assert result["values"]["schedules"][0]["outcome_count"] == 1
    assert result["acceptance_claim"] == "not_established"


def test_rehashed_wrong_online_decision_does_not_pass_first_match_reference(online_originals):
    _, path, _, binding = online_originals
    with sqlite3.connect(path / "workspace/marvis.sqlite") as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM reference_decisions WHERE request_id='req-0'").fetchone()
        response = json.loads(row["response_json"])
        response["action"]["type"] = "review"
        response["output_hash"] = digest({key: response[key] for key in ("action", "matched_rule_id", "score", "score_product", "package_hash", "versions", "status")})
        conn.execute("DROP TRIGGER reference_decision_completed_immutable")
        conn.execute("UPDATE reference_decisions SET response_json=? WHERE request_id='req-0'", (json.dumps(response),))
        revised = dict(conn.execute("SELECT * FROM reference_decisions WHERE request_id='req-0'").fetchone())
    sha = repin_synthetic_database(path)
    first = binding.decisions[0].model_copy(update={"record_sha256": digest(revised)})
    result = revalidate_online_archive(path, expected_manifest_sha256=sha, frozen_binding=binding.model_copy(update={"decisions": [first, *binding.decisions[1:]]}))
    assert result["checks"]["archive_integrity"]["status"] == "verified"
    assert result["checks"]["independent_decisions"]["status"] == "mismatch"
    assert not result["supported_checks_verified"]


def test_extension_inputs_reject_routes_keys_unchecked_network_and_wrong_roles():
    adapter = TypeAdapter(ExtensionBusinessInput)
    for value in ({"kind": "arbitrary_http", "actor": "maker", "route": "/api/unreviewed"}, {"kind": "register_historical_source_profile", "actor": "admin", "request": {"profile_id": "p", "mode": "reference_http", "endpoint": "http://127.0.0.1:1234"}}, {"kind": "execute_risk_source_query", "actor": "maker", "request_id": "one", "signing_key": "secret"}, {"kind": "execute_risk_source_query", "actor": "admin", "request_id": "one"}):
        with pytest.raises(ValidationError):
            adapter.validate_python(value)


def test_source_assessment_keeps_unknown_debt_and_partial_coverage():
    from marvis.risk_context.source_contracts import SourceEnvelope, assess, instant
    value = SourceEnvelope(subject_namespace="fixture", subject_token="a" * 64, record_version=1, observed_at="2026-08-01T00:00:00Z", available_at="2026-08-01T00:01:00Z", expires_at="2026-09-01T00:00:00Z", bureau={"coverage": "complete", "accounts": [{"account_token": "b" * 64, "currency": "CNY", "balance_minor": None, "past_due_minor": 25}]})
    assert source_assessment_reference(value.model_dump(), "2026-08-02T00:00:00Z") == assess(value, instant("2026-08-02T00:00:00Z"))
    assert source_assessment_reference(value.model_dump(), "2026-08-02T00:00:00Z")["totals_by_currency"]["CNY"]["balance_minor"] is None


@pytest.mark.parametrize("observed", [True, 1.0])
def test_measured_count_and_money_preserve_integer_type(observed):
    with pytest.raises(ValueError, match="exact integer"):
        measured_values({"amount_minor": observed}, {"amount_minor": 1})


def registered(rt, name, frame):
    path = rt.materials / name
    frame.to_parquet(path, index=False)
    data = rt.app.state.risk_events.repo.registry.register_existing(path, task_id=rt.task.id, role="event_history")
    return data, {"path": path.name, "sha256": digest(path.read_bytes()), "source_kind": "synthetic"}


@pytest.fixture
def connector_originals(tmp_path):
    rt = native(tmp_path)
    source_prefix = f"/api/tasks/{rt.task.id}/risk-sources"
    basis = {"basis_id": "basis", "reference": "synthetic-local-source", "declaration": "synthetic_reference_test"}
    basis_artifact = require(rt.admin.post(source_prefix + "/authorization-bases", json=basis), 201)
    profile = {"profile_id": "history", "mode": "historical_file"}
    require(rt.admin.post("/api/risk-sources/profiles", json={"profile": profile}), 201)
    now = datetime.now(UTC)
    grant = {"grant_id": "source-grant", "profile_id": "history", "task_id": rt.task.id, "grantee_id": rt.actors["maker"], "subject_namespace": "fixture", "subject_token": "a" * 64, "purpose": "credit_review", "basis_artifact_id": basis_artifact["id"], "starts_at": (now - timedelta(hours=1)).isoformat(), "expires_at": (now + timedelta(hours=1)).isoformat()}
    require(rt.admin.post("/api/risk-sources/grants", json=grant), 201)
    record = {"subject_namespace": "fixture", "subject_token": "a" * 64, "record_version": 1, "observed_at": (now - timedelta(minutes=3)).isoformat(), "available_at": (now - timedelta(minutes=2)).isoformat(), "expires_at": (now + timedelta(hours=1)).isoformat(), "bureau": {"coverage": "partial", "accounts": []}}
    data, material = registered(rt, "source.parquet", pd.DataFrame({"id": ["record-1"], "response": [json.dumps(record)]}))
    req = {"request_id": "source-one", "grant_id": grant["grant_id"], "historical": {"dataset_id": data.id, "expected_content_hash": data.content_hash, "row_id_column": "id", "row_id": "record-1", "response_column": "response"}}
    require(rt.maker.post(source_prefix + "/requests", json=req), 201)
    result = require(rt.maker.post(source_prefix + "/requests/source-one/execute"), 200)
    details = require(rt.maker.get(result["evidence_url"]), 200)
    assert details["assessment"]["account_count"] is None
    actions = [{"kind": "declare_source_authorization_basis", "actor": "admin", "request": basis}, {"kind": "register_historical_source_profile", "actor": "admin", "request": profile}, {"kind": "grant_risk_source", "actor": "admin", "request": grant}, {"kind": "prepare_risk_source_query", "actor": "maker", "request": req}, {"kind": "execute_risk_source_query", "actor": "maker", "request_id": "source-one"}]
    case = case_for(rt, "connector_antifraud", actions, [material])
    native_row = rows(rt, "SELECT * FROM source_requests WHERE task_id=? AND id='source-one'", (rt.task.id,))[0]
    artifact = rows(rt, "SELECT * FROM task_artifacts WHERE id=?", (result["artifact_id"],))[0]
    binding = FrozenConnectorBinding(**common(rt, case), source_requests=[{"request": req, "record_sha256": digest(native_row), "artifact_id": result["artifact_id"], "artifact_sha256": artifact["content_hash"]}])
    path, sha = archive(rt, binding)
    return rt, path, sha, binding


def test_real_http_toolrunner_history_source_archive_uses_exact_original_cell(connector_originals, monkeypatch):
    _, path, sha, binding = connector_originals
    monkeypatch.setattr("marvis.risk_context.source_contracts.assess", lambda *a, **k: pytest.fail("producer used as proof"))
    show_synthetic_reference_failure(monkeypatch)
    result = revalidate_connector_archive(path, expected_manifest_sha256=sha, frozen_binding=binding)
    assert result["supported_checks_verified"], result["checks"]
    assert result["values"]["sources"][0]["account_count"] is None
    assert result["values"]["sources"][0]["totals_by_currency"] is None
    assert not (path / "workspace/source_private").exists()


@pytest.fixture
def event_originals(tmp_path):
    rt = native(tmp_path)
    src = source(task_id=rt.task.id)
    basis_input = {"basis_id": "basis", "reference": "synthetic-event-history", "declaration": "synthetic_reference_test"}
    basis = require(rt.admin.post(f"/api/tasks/{rt.task.id}/risk-sources/authorization-bases", json=basis_input), 201)
    require(rt.admin.post("/api/risk-events/sources", json=src.model_dump()), 201)
    now = datetime.now(UTC)
    grant = {"grant_id": "event-grant", "task_id": rt.task.id, "source_id": src.source_id, "source_contract_hash": src.contract_hash, "grantee_id": rt.actors["maker"], "permissions": ["read", "write"], "purpose": "synthetic event declarations", "basis_artifact_id": basis["id"], "starts_at": (now - timedelta(hours=1)).isoformat(), "expires_at": (now + timedelta(hours=1)).isoformat()}
    require(rt.admin.post("/api/risk-events/grants", json=grant), 201)
    from test_event_features import identity
    material_list, import_actions = [], []
    for kind, claims in (("events", [event(values={"amount_minor": 2**53 + 1}), event("evt-2", values={"amount_minor": 7}), event("evt-3", subject=identity(name="bob"))]), ("coverage", [coverage()])):
        data, material = registered(rt, kind + ".parquet", pd.DataFrame({"claim": [json.dumps(row.model_dump()) for row in claims]}))
        material_list.append(material)
        payload = {"grant_id": grant["grant_id"], "dataset": {"source_id": src.source_id, "source_contract_hash": src.contract_hash, "dataset_id": data.id, "expected_content_hash": data.content_hash, "kind": kind, "json_column": "claim"}}
        require(rt.maker.post(f"/api/tasks/{rt.task.id}/risk-events/imports", json=payload), 201)
        import_actions.append({"kind": "import_risk_event_dataset", "actor": "maker", "request": payload})
    contract = query(src, knowledge_cutoff=datetime.now(UTC).isoformat(), availability_mode="retrospective_declared")
    req = {"request_id": "event-one", "grant_id": grant["grant_id"], "contract": contract.model_dump()}
    proposal = require(rt.maker.post(f"/api/tasks/{rt.task.id}/risk-events/requests", json=req), 201)
    current = approve(rt, gated_plan(rt, proposal))
    assert current["status"] == "done", current
    saved = require(rt.maker.get(f"/api/tasks/{rt.task.id}/risk-events/requests/event-one/evidence", params={"grant_id": grant["grant_id"]}), 200)
    intent_row = rows(rt, "SELECT * FROM event_requests WHERE task_id=? AND request_id='event-one'", (rt.task.id,))[0]
    decision_row = rows(rt, "SELECT * FROM event_decisions WHERE task_id=?", (rt.task.id,))[0]
    artifact = rows(rt, "SELECT * FROM task_artifacts WHERE id=?", (saved["artifact_id"],))[0]
    actions = [{"kind": "declare_source_authorization_basis", "actor": "admin", "request": basis_input}, {"kind": "register_risk_event_source", "actor": "admin", "request": src.model_dump()}, {"kind": "grant_risk_event_source", "actor": "admin", "request": grant}, *import_actions, {"kind": "prepare_risk_event_features", "actor": "maker", "request": req}]
    case = case_for(rt, "connector_antifraud", actions, material_list)
    binding = FrozenConnectorBinding(**common(rt, case), event_requests=[{"request": req, "intent_record_sha256": digest(intent_row), "decision_record_sha256": digest(decision_row), "artifact_id": artifact["id"], "artifact_sha256": artifact["content_hash"]}])
    path, sha = archive(rt, binding)
    return rt, path, sha, binding


def test_real_approved_isolated_event_archive_recomputes_large_integer_and_graph(event_originals, monkeypatch):
    _, path, sha, binding = event_originals
    monkeypatch.setattr("marvis.risk_context.event_replay.evaluate_event_snapshot", lambda *a, **k: pytest.fail("producer used as proof"))
    show_synthetic_reference_failure(monkeypatch)
    result = revalidate_connector_archive(path, expected_manifest_sha256=sha, frozen_binding=binding)
    assert result["supported_checks_verified"], result["checks"]
    values = result["values"]["events"][0]["features"]
    assert values["amount"]["value"] == 2**53 + 8
    assert values["transactions"]["value"] == 2
    assert values["shared_devices"]["value"] == 1
    assert result["values"]["events"][0]["fraud_or_identity_proof"] is False


@pytest.mark.parametrize("changes", [{}, {"available_at": None}, {"values": {}}, {"version": 3, "supersedes_version": 2}])
def test_independent_event_unknown_availability_missing_numeric_and_lineage(changes):
    from test_event_features import snapshot
    from marvis.risk_context.event_replay import evaluate_event_snapshot
    snap = snapshot([event(**changes)])
    assert event_features_reference(snap.model_dump()) == evaluate_event_snapshot(snap)


def test_rehashed_event_metric_is_rejected_by_independent_window_math(event_originals):
    _, path, _, binding = event_originals
    with sqlite3.connect(path / "workspace/marvis.sqlite") as conn:
        conn.row_factory = sqlite3.Row
        native = conn.execute("SELECT * FROM event_decisions").fetchone()
        value = json.loads(native["payload"])
        value["result"]["features"]["amount"]["value"] += 99
        conn.execute("DROP TRIGGER event_decisions_immutable")
        conn.execute("UPDATE event_decisions SET payload=?,receipt_hash=?", (json.dumps(value), digest(value)))
        revised = dict(conn.execute("SELECT * FROM event_decisions").fetchone())
    sha = repin_synthetic_database(path)
    item = binding.event_requests[0].model_copy(update={"decision_record_sha256": digest(revised)})
    result = revalidate_connector_archive(path, expected_manifest_sha256=sha, frozen_binding=binding.model_copy(update={"event_requests": [item]}))
    assert result["checks"]["archive_integrity"]["status"] == "verified"
    assert result["checks"]["independent_event_features"]["status"] == "mismatch"
    assert not result["supported_checks_verified"]


def test_real_http_approved_collection_queue_and_final_effect_closure(tmp_path, monkeypatch):
    from test_collection_runtime import runtime, plan, approve as approve_collection
    rt = runtime.__wrapped__(tmp_path)
    rt.root = tmp_path
    rt.materials = rt.settings.tasks_dir / rt.task.id / "materials"
    rt.materials.mkdir()
    assert approve_collection(rt, plan(rt, "queue_batch", "催收参考排队"))["status"] == "done"
    assert approve_collection(rt, plan(rt, "execute_reference", "催收参考执行"))["status"] == "done"
    record = rows(rt, "SELECT * FROM collection_batches WHERE task_id=? AND batch_id='http-batch'", (rt.task.id,))[0]
    frozen_records = {"batch": record}
    for table in ("collection_queue_items", "collection_native_effects", "collection_action_finals"):
        frozen_records[table] = sorted(rows(rt, f"SELECT * FROM {table} WHERE task_id=?", (rt.task.id,)), key=digest)
    case = case_for(rt, "collections", [{"kind": "prepare_collection_batch", "actor": "maker", "request": json.loads(record["request_json"])}])
    binding = FrozenCollectionsBinding(**common(rt, case), batches=[{"batch_id": "http-batch", "records_sha256": digest(frozen_records)}])
    path, sha = archive(rt, binding)
    monkeypatch.setattr("marvis.collection.execution.CollectionExecutor.apply", lambda *a, **k: pytest.fail("producer used as proof"))
    show_synthetic_reference_failure(monkeypatch)
    result = revalidate_collections_archive(path, expected_manifest_sha256=sha, frozen_binding=binding)
    assert result["supported_checks_verified"], result["checks"]
    assert result["values"]["batches"] == [{"batch_id": "http-batch", "status": "completed", "revision": 3, "action_count": 1, "customer_contacted": False, "actual_cost_minor": None}]
