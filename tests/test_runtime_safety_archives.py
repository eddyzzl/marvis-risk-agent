"""Synthetic local checks on actual product requests and original native bytes."""
from contextlib import closing
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
import sqlite3
import time

from fastapi.testclient import TestClient
import pytest
from pydantic import ValidationError

from marvis.app import create_app
from marvis.db import StrategyRepository
from marvis.db_schema import connect
from marvis.orchestrator.eval.runtime_archive_safety import revalidate_authorization_archive, revalidate_duplicate_effects_archive, revalidate_point_in_time_archive, revalidate_recovery_archive
from marvis.orchestrator.eval.runtime_archive_online import FrozenOnlineBinding, revalidate_online_archive
from marvis.orchestrator.eval.runtime_archive_connector import FrozenConnectorBinding, revalidate_connector_archive
from marvis.orchestrator.eval.runtime_archive_collections import FrozenCollectionsBinding, revalidate_collections_archive
from marvis.orchestrator.eval.runtime_contracts import RuntimeAction, RuntimeCase, digest
from marvis.orchestrator.eval.runtime_custody import CustodyError, capture_case, prepare_custody_run, verify_case_archive
from marvis.orchestrator.eval.runtime_extensions import submit_extension, start_extension_workflow, extension_step_decision
from marvis.orchestrator.eval.runtime_extensions_archive import ExtensionArchive
from marvis.orchestrator.eval.runtime_native_identity import declaration_matches
from marvis.orchestrator.eval.runtime_runner import Journey, _receipts
from marvis.orchestrator.eval.runtime_safety_capture import capture_runtime_before
from marvis.orchestrator.eval.runtime_safety_contracts import FrozenSafetyBinding
from marvis.packs.strategy.dsl import StrategySpec, StrategyRuleSpec, StrategyAction
from marvis.packs.strategy.strategy import build_strategy_from_spec
from marvis.repositories.tasks import TaskRepository
from tests.test_db import _task_create
from tests.test_runtime_evidence_custody import simple_capture
from tests.test_runtime_extension_journeys import claim, binding, material_refs
from tests.test_runtime_extensions_archives import common, repin_synthetic_database, show_synthetic_reference_failure


def request(kind, value=None, *, actor="maker", bindings=(), **fields):
    item = {"kind": kind, "actor": actor, **fields}
    if value is not None:
        item["request"] = value
    return RuntimeAction(kind="submit_extension_request", content="synthetic complete public business input", extension_request=item, extension_bindings=list(bindings))


def local_run(tmp_path, family, actions, *, prerequisite=None, materials=()):
    app = create_app(tmp_path / "workspace")
    task = TaskRepository(app.state.settings.db_path).create_task(_task_create(task_type="strategy", run_mode="manual"))
    if prerequisite:
        prerequisite(app, task)
    case = RuntimeCase(id=family, revision="synthetic-original-safety", family=family, task={"task_type": "strategy"}, actions=actions, materials=list(materials), business_constraints_source="synthetic local reference checks; no hidden acceptance values")
    before = tmp_path / "before"
    origin_ns = time.monotonic_ns()
    receipt = capture_runtime_before(before, app.state.settings.workspace, case, deadline=time.monotonic() + 30, origin_ns=origin_ns)
    original_hash = digest(case.model_dump())
    with TestClient(app) as client:
        journey = Journey(client, case, time.monotonic() + 120, origin_ns=origin_ns)
        journey.task_id = task.id
        journey.workspace = app.state.settings.workspace
        for material in materials:
            path = tmp_path / "materials" / material["path"]
            with path.open("rb") as stream:
                uploaded = journey.json_request("POST", f"/api/tasks/{task.id}/datasets/upload", label="upload_material", files={"file": (path.name, stream)}, data={"role": "unknown"})
            journey.uploaded_materials.append({"datasets": uploaded["datasets"]})
        for index, action in enumerate(case.actions, 1):
            journey.current_action_index = index
            if action.kind == "submit_extension_request":
                submit_extension(journey, action, app.state.settings.workspace)
            elif action.kind == "start_extension_workflow":
                start_extension_workflow(journey, action)
            elif action.kind == "recover_governed_decision":
                from marvis.orchestrator.eval.runtime_fault_capture import recover_fault
                recover_fault(journey, action, journey.workspace)
            else:
                assert extension_step_decision(journey, action)
        execution, private = _receipts(app.state.settings.workspace, task.id, case=case, journey=journey)
    private.update(extension_outputs=journey.extension_outputs, extension_decisions=getattr(journey, "extension_decisions", []))
    for key in ("human_decisions", "runtime_faults"):
        if getattr(journey, key, None):
            private[key] = getattr(journey, key)
    rt = type("LocalOriginals", (), {"task": task, "root": tmp_path, "settings": app.state.settings, "materials": tmp_path / "materials"})()
    rt.materials.mkdir(exist_ok=True)
    return rt, case, journey, execution, private, before, receipt, original_hash


def retain(rt, frozen, execution, private, before):
    run = prepare_custody_run(rt.root / "private", frozen.run_id, forbidden=(rt.materials,))
    metadata = {key: getattr(frozen, key) for key in ("run_id", "cases_sha256", "expected_sha256", "source", "model_connection_sha256")}
    metadata["execution_before_scoring"] = {"execution": execution, "http_events": private.pop("http_events")}
    receipt = capture_case(run_dir=run, workspace=rt.settings.workspace, dataset_root=rt.materials, case=frozen.case, private=private, bindings=metadata, owned_processes_stopped=True, before_dir=before)
    return run / frozen.case.id, receipt["manifest_sha256"]


def test_future_before_snapshot_is_original_private_and_cannot_be_replaced(tmp_path):
    args = simple_capture(tmp_path)
    before = tmp_path / "before"
    receipt = capture_runtime_before(before, args["workspace"], args["case"], deadline=time.monotonic() + 5, origin_ns=time.monotonic_ns())
    with closing(sqlite3.connect(args["workspace"] / "marvis.sqlite")) as conn:
        conn.execute("UPDATE records SET value=0.91")
        conn.commit()
    final = capture_case(**args, before_dir=before)
    archive = args["run_dir"] / args["case"].id
    assert verify_case_archive(archive, expected_manifest_sha256=final["manifest_sha256"])["archived_bytes_match_manifest"]
    assert json.loads((archive / "manifest.json").read_text())["safety_before"]["manifest_sha256"] == receipt["manifest_sha256"]
    for carrier, expected in (("safety/before", .73), ("workspace", .91)):
        with closing(sqlite3.connect(archive / carrier / "marvis.sqlite")) as conn:
            assert conn.execute("SELECT value FROM records").fetchone()[0] == expected
    with pytest.raises(CustodyError, match="collision"):
        capture_runtime_before(before, args["workspace"], args["case"], deadline=time.monotonic() + 5, origin_ns=time.monotonic_ns())
    (archive / "safety/before/marvis.sqlite").write_bytes(b"tampered")
    with pytest.raises(CustodyError):
        verify_case_archive(archive, expected_manifest_sha256=final["manifest_sha256"])


@pytest.mark.parametrize("kind", ["deadline", "credential", "alias"])
def test_before_capture_refuses_budget_credentials_and_aliases(tmp_path, kind):
    args = simple_capture(tmp_path)
    workspace = args["workspace"]
    if kind == "credential":
        with closing(sqlite3.connect(workspace / "marvis.sqlite")) as conn:
            conn.execute("CREATE TABLE private(api_key TEXT)")
            conn.execute("INSERT INTO private VALUES('synthetic-secret')")
            conn.commit()
    if kind == "alias":
        alias = tmp_path / "alias"
        alias.symlink_to(workspace, target_is_directory=True)
        workspace = alias
    with pytest.raises(CustodyError):
        capture_runtime_before(tmp_path / "before", workspace, args["case"], deadline=time.monotonic() + (-1 if kind == "deadline" else 5), origin_ns=time.monotonic_ns())


@pytest.fixture
def online_originals(tmp_path):
    def prerequisite(app, task):
        spec = StrategySpec(strategy_type="approval", rules=(StrategyRuleSpec(rule_id="low", priority=1, condition={"op": "compare", "field": "x1", "operator": "<", "value": .5}, action=StrategyAction(type="approval", reason_code="LOW")),), default_action=StrategyAction(type="review", reason_code="REVIEW"))
        strategy = replace(build_strategy_from_spec(spec, score_col="x1", description="synthetic prerequisite only"), id="reference-strategy")
        StrategyRepository(app.state.settings.db_path).create_strategy(task.id, strategy)
        with connect(app.state.settings.db_path) as conn:
            conn.execute("UPDATE strategies SET asset_status='validated' WHERE id=?", (strategy.id,))
    actions = [claim(role) for role in ("maker", "checker", "admin")]
    actions += [request("build_decision_package", {"package_kind": "rule_only", "strategy_id": "reference-strategy", "strategy_version": 1, "decision_node": "underwriting", "raw_schema": [{"name": "x1", "type": "number"}]}),
                request("request_production_promotion", {"environment": "local-reference", "deployment_slot": "production", "strategy_id": "reference-strategy", "strategy_version": 1, "decision_package_hash": "0" * 64, "reason": "synthetic complete package"}, bindings=[binding("request.decision_package_hash", 4, "package_hash")])]
    actions += [request("review_production_promotion", {"decision": "approve", "reason": "synthetic independent review"}, actor=role, promotion_id="placeholder", bindings=[binding("promotion_id", 5)]) for role in ("checker", "admin")]
    actions += [request("install_decision_package", {"promotion_id": "placeholder", "probe_features": {"x1": .1}}, actor="admin", bindings=[binding("request.promotion_id", 5)]), request("activate_production_promotion", {"verifier_id": "local-reference.v1", "receipt_id": "placeholder", "reason": "actual installation"}, actor="admin", promotion_id="placeholder", bindings=[binding("promotion_id", 5), binding("request.receipt_id", 8, "activation_evidence.receipt_id")])]
    query = {"request_id": "duplicate-decision", "decision_node": "underwriting", "expected_package_hash": "0" * 64, "features": {"x1": .1}}
    actions += [request(kind, query, bindings=[binding("request.expected_package_hash", 4, "package_hash")]) for kind in ("submit_reference_decision", "replay_reference_decision")]
    rt, case, journey, execution, private, before, captured, case_hash = local_run(tmp_path, "online_decision", actions, prerequisite=prerequisite)
    private["http_events"] = journey.events
    with connect(rt.settings.db_path) as conn:
        decision = dict(conn.execute("SELECT * FROM reference_decisions").fetchone())
    frozen = FrozenSafetyBinding(**common(rt, case), before_manifest_sha256=captured["manifest_sha256"], duplicate_operations=[{"kind": "reference_decision", "action_indices": [10, 11], "request_id": query["request_id"], "slot": "production", "record_sha256": digest(decision)}])
    archive, sha = retain(rt, frozen, execution, private, before)
    return rt, archive, sha, frozen, case_hash


def test_original_placeholders_are_resolved_only_from_native_rows_and_duplicate_is_one(online_originals, monkeypatch):
    _, archive, sha, binding, original_hash = online_originals
    show_synthetic_reference_failure(monkeypatch)
    result = revalidate_duplicate_effects_archive(archive, expected_manifest_sha256=sha, frozen_binding=binding)
    assert result["status"] == "verified", result["checks"]
    assert result["values"]["operations"][0]["durable_completed_rows"] == 1
    assert digest(binding.case.model_dump()) == original_hash
    assert binding.case.actions[9].extension_request.request.expected_package_hash == "0" * 64
    monkeypatch.undo()
    missing = binding.model_copy(update={"before_manifest_sha256": None})
    assert revalidate_duplicate_effects_archive(archive, expected_manifest_sha256=sha, frozen_binding=missing)["status"] == "unavailable"
    assert revalidate_recovery_archive(archive, expected_manifest_sha256=sha, frozen_binding=binding)["status"] == "unavailable"


def test_native_identity_row_tamper_is_not_blessed_by_new_manifest_pin(online_originals):
    _, archive, _, binding, _ = online_originals
    with closing(sqlite3.connect(archive / "workspace/marvis.sqlite")) as conn:
        conn.execute("UPDATE production_principals SET role='admin' WHERE role='maker'")
        conn.commit()
    sha = repin_synthetic_database(archive)
    result = revalidate_duplicate_effects_archive(archive, expected_manifest_sha256=sha, frozen_binding=binding)
    assert result["status"] == "mismatch", result["checks"]


def test_frozen_request_structural_admission_keeps_all_unbound_business_values(online_originals):
    _, archive, sha, frozen, _ = online_originals
    with ExtensionArchive(archive, sha, frozen, "synthetic", ()) as ctx:
        ctx.open()
        actual = ctx.business_case.actions[9].request
        assert declaration_matches(frozen.case, "submit_reference_decision", actual)
        changed = actual.model_copy(update={"features": {"x1": .9}})
        assert not declaration_matches(frozen.case, "submit_reference_decision", changed)


def test_online_math_reader_accepts_original_typed_placeholder_case(online_originals):
    rt, archive, sha, frozen, _ = online_originals
    private = json.loads((archive / "private.json").read_bytes())
    outputs = private["extension_outputs"]
    package, deployment_id = outputs[3]["response"]["package_hash"], outputs[8]["response"]["id"]
    with connect(rt.settings.db_path) as conn:
        def one(sql, params=()):
            return dict(conn.execute(sql, params).fetchone())
        deployed = one("SELECT * FROM production_deployments WHERE id=?", (deployment_id,))
        promotion = one("SELECT * FROM production_promotion_requests WHERE id=?", (deployed["promotion_request_id"],))
        records = {"deployment": deployed, "promotion": promotion,
                   "approvals": [dict(r) for r in conn.execute("SELECT * FROM production_promotion_approvals WHERE request_id=? ORDER BY stage", (promotion["id"],))],
                   "installation": one("SELECT * FROM reference_installations WHERE promotion_id=?", (promotion["id"],)),
                   "manifest": one("SELECT * FROM production_deployment_manifests WHERE id=?", (deployed["manifest_hash"],)),
                   "head": one("SELECT * FROM production_environment_heads WHERE environment='local-reference'")}
        native_package = one("SELECT * FROM reference_decision_packages WHERE id=?", (package,))
        audit = one("SELECT event_hash FROM production_governance_events ORDER BY sequence DESC LIMIT 1")["event_hash"]
    binding = FrozenOnlineBinding(**{key: getattr(frozen, key) for key in ("case", "run_id", "task_id", "cases_sha256", "expected_sha256", "source", "model_connection_sha256")}, package_hash=package, package_record_sha256=digest(native_package), deployment_id=deployment_id, deployment_records_sha256=digest(records), production_audit_head_sha256=audit, decisions=[{"request": outputs[9]["request"]["request"], "slot": "production", "record_sha256": frozen.duplicate_operations[0].record_sha256}])
    result = revalidate_online_archive(archive, expected_manifest_sha256=sha, frozen_binding=binding)
    assert result["supported_checks_verified"], result["checks"]


def make_governed(tmp_path, *, reject=False, fault=None):
    from tests.test_collection_cashflows import UNIT
    from tests.test_collection_planning import AT, case, history, policy, spec
    from marvis.collection.actions import CollectionPolicy
    refs = {"source_artifact_id": "placeholder", "source_artifact_hash": "0" * 64}
    p = policy(unit=UNIT, basis_artifact_id="placeholder", basis_artifact_hash="0" * 64, max_estimated_active_cost_minor=100)
    bounded = p.model_dump()
    bounded["queues"][0]["max_active_actions"] = 5
    p = CollectionPolicy.model_validate(bounded)
    actions = [claim(role) for role in ("maker", "checker", "admin")]
    actions += [request("declare_collection_material", {"material_id": "declared", "description": "synthetic private source declarations", "declarations": {"synthetic": True}}),
                request("create_collection_case", {"case_id": "case-1", "subject_namespace": "fixture", "subject_token": "a" * 64, "unit": UNIT, "opening_balance_minor": 10000, "opened_at": "2026-08-01T00:00:00Z", "source_assurance": "historical_import_unverified", **refs}, bindings=material_refs("request.source_artifact"))]
    batch = {"batch_id": "governed-batch", "execution_mode": "local_reference", "strategy": spec(p).to_dict(), "policy": p.model_dump(), "cases": [case("case-1", subject_namespace="fixture", **refs).model_dump()], "histories": [history(subject_namespace="fixture", **refs).model_dump()], "as_of": AT, "knowledge_cutoff": AT}
    actions += [request("prepare_collection_batch", batch, bindings=[*material_refs("request.policy.basis_artifact", "request.cases[].source_artifact", "request.histories[].source_artifact"), binding("request.strategy.default_action.value.policy_hash", source_field="resolved_policy_hash")])]
    for tool in (("collection.queue_batch",) if reject else ("collection.queue_batch", "collection.execute_reference")):
        actions += [RuntimeAction(kind="start_extension_workflow", tool=tool, content="synthetic start of exact native proposal", extension_source_action=6)]
        if fault and tool == "collection.queue_batch":
            actions += [RuntimeAction(kind="approve_step", tool=tool, content="synthetic independent exact maker/checker review", fault={"kind": fault})]
            if fault == "discard_success_response_once":
                actions += [RuntimeAction(kind="recover_governed_decision", tool=tool, content="synthetic explicit original-target readback", fault_source_action=len(actions))]
            else:
                actions += [RuntimeAction(kind="approve_step", tool=tool, content="synthetic separate later explicit approval")]
        else:
            actions += [RuntimeAction(kind="reject_step" if reject else "approve_step", tool=tool, content="synthetic independent exact maker/checker review")]
    rt, case, journey, execution, private, before, captured, _ = local_run(tmp_path, "collections", actions)
    private["http_events"] = journey.events
    identities = []
    with connect(rt.settings.db_path) as conn:
        for capture in private["extension_decisions"] + private.get("human_decisions", []):
            row = dict(conn.execute("SELECT * FROM decision_records WHERE id=?", (capture["response"]["decision_id"],)).fetchone())
            identities.append({"action_index": capture["case_action_index"], "plan_id": row["plan_id"], "step_id": row["step_id"], "decision_id": row["id"], "decision_record_sha256": digest(row)})
    frozen = FrozenSafetyBinding(**common(rt, case), before_manifest_sha256=captured["manifest_sha256"], governed_actions=identities,
                                boundary_action_indices=[8] if fault == "stale_step_cas_once" else [],
                                recovery_action_indices=[9] if fault == "discard_success_response_once" else [])
    archive, sha = retain(rt, frozen, execution, private, before)
    return archive, sha, frozen


@pytest.fixture
def governed_originals(tmp_path):
    return make_governed(tmp_path)


def test_authorization_recomputes_original_cas_actor_and_one_shot_effect(governed_originals, monkeypatch):
    archive, sha, binding = governed_originals
    show_synthetic_reference_failure(monkeypatch)
    result = revalidate_authorization_archive(archive, expected_manifest_sha256=sha, frozen_binding=binding)
    assert result["status"] == "verified", result["checks"]
    assert [r["tool"] for r in result["values"]["governed_actions"]] == ["collection.queue_batch", "collection.execute_reference"]
    assert result["source_authentication"] == result["acceptance_claim"] == "not_established"


def test_authorization_requires_real_native_rejection_and_no_invocation(tmp_path):
    archive, sha, binding = make_governed(tmp_path, reject=True)
    result = revalidate_authorization_archive(archive, expected_manifest_sha256=sha, frozen_binding=binding)
    assert result["status"] == "verified", result["checks"]
    assert result["values"]["governed_actions"][0]["decision"] == "reject"


def test_collection_math_reader_accepts_original_artifact_policy_refs(governed_originals):
    archive, sha, frozen = governed_originals
    with closing(sqlite3.connect(archive / "workspace/marvis.sqlite")) as conn:
        conn.row_factory = sqlite3.Row
        batch = dict(conn.execute("SELECT * FROM collection_batches").fetchone())
        records = {"batch": batch}
        for table in ("collection_queue_items", "collection_native_effects", "collection_action_finals"):
            records[table] = sorted([dict(row) for row in conn.execute(f"SELECT * FROM {table}")], key=digest)
    binding = FrozenCollectionsBinding(**{key: getattr(frozen, key) for key in ("case", "run_id", "task_id", "cases_sha256", "expected_sha256", "source", "model_connection_sha256")}, batches=[{"batch_id": batch["batch_id"], "records_sha256": digest(records)}])
    result = revalidate_collections_archive(archive, expected_manifest_sha256=sha, frozen_binding=binding)
    assert result["supported_checks_verified"], result["checks"]


def repin_private(archive, private):
    file = archive / "private.json"
    file.write_text(json.dumps(private, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    manifest_path = archive / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["files"]["private.json"] = {"sha256": digest(file.read_bytes()), "size_bytes": file.stat().st_size}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return digest(manifest_path.read_bytes())


@pytest.mark.parametrize("mutation", ["cas", "actor", "missing_before"])
def test_repin_cannot_bless_missing_or_modified_original_human_cas(governed_originals, mutation):
    archive, _, binding = governed_originals
    private = json.loads((archive / "private.json").read_bytes())
    capture = private["extension_decisions"][0]
    if mutation == "cas":
        capture["plan_before"]["steps"][0]["inputs"]["batch_id"] = "different-batch"
    elif mutation == "actor":
        capture["actor"] = "maker"
    else:
        capture.pop("plan_before")
    sha = repin_private(archive, private)
    result = revalidate_authorization_archive(archive, expected_manifest_sha256=sha, frozen_binding=binding)
    assert result["status"] == ("unavailable" if mutation == "missing_before" else "mismatch"), result["checks"]


@pytest.fixture
def event_originals(tmp_path):
    import pandas as pd
    from tests.test_event_features import source, event, coverage, query
    src = source(task_id="placeholder")
    now = datetime.now(UTC)
    actions = [claim(role) for role in ("maker", "checker", "admin")]
    actions += [request("declare_source_authorization_basis", {"basis_id": "event-basis", "reference": "synthetic-local-time-check", "declaration": "synthetic_reference_test"}, actor="admin"), request("register_risk_event_source", src.model_dump(), actor="admin", bindings=[binding("request.task_id", source_field="task_id")])]
    grant = {"grant_id": "event-grant", "task_id": "placeholder", "source_id": src.source_id, "source_contract_hash": "0" * 64, "grantee_id": "placeholder", "permissions": ["read", "write"], "purpose": "synthetic event time check", "basis_artifact_id": "placeholder", "starts_at": (now - timedelta(hours=1)).isoformat(), "expires_at": (now + timedelta(hours=1)).isoformat()}
    actions += [request("grant_risk_event_source", grant, actor="admin", bindings=[binding("request.task_id", source_field="task_id"), binding("request.grantee_id", 1), binding("request.basis_artifact_id", 4), binding("request.source_contract_hash", 5, "contract_hash")])]
    directory = tmp_path / "materials"
    directory.mkdir()
    materials = []
    for index, (kind, claims) in enumerate((("events", [event(values={"amount_minor": 2**53 + 1}), event("second", values={"amount_minor": 7})]), ("coverage", [coverage()])), 1):
        path = directory / (kind + ".parquet")
        pd.DataFrame({"claim": [json.dumps(row.model_dump()) for row in claims]}).to_parquet(path, index=False)
        materials.append({"path": path.name, "sha256": digest(path.read_bytes()), "role": "unknown", "source_kind": "synthetic"})
        value = {"grant_id": "event-grant", "dataset": {"source_id": src.source_id, "source_contract_hash": "0" * 64, "dataset_id": "placeholder", "expected_content_hash": "0" * 64, "kind": kind, "json_column": "claim"}}
        actions += [request("import_risk_event_dataset", value, bindings=[binding("request.dataset.source_contract_hash", 5, "contract_hash"), binding("request.dataset.dataset_id", source_field="dataset_id", source_material=index), binding("request.dataset.expected_content_hash", source_field="dataset_content_hash", source_material=index)])]
    value = {"request_id": "time-event", "grant_id": "event-grant", "contract": query(src, availability_mode="retrospective_declared").model_dump()}
    actions += [request("prepare_risk_event_features", value, bindings=[binding("request.contract.source_contract_hash", 5, "contract_hash"), binding("request.contract.knowledge_cutoff", 8, "ingested_at")]), RuntimeAction(kind="start_extension_workflow", content="synthetic exact event proposal", tool="risk_context.replay_events", extension_source_action=9), RuntimeAction(kind="approve_step", content="synthetic exact event grant and cutoff review", tool="risk_context.replay_events"), request("read_risk_event_evidence", request_id="placeholder", grant_id="event-grant", bindings=[binding("request_id", 9, "request_id")])]
    rt, case, journey, execution, private, before, captured, _ = local_run(tmp_path, "connector_antifraud", actions, materials=materials)
    private["http_events"] = journey.events
    with connect(rt.settings.db_path) as conn:
        intent = dict(conn.execute("SELECT * FROM event_requests WHERE task_id=?", (rt.task.id,)).fetchone())
        record = dict(conn.execute("SELECT * FROM event_decisions WHERE task_id=?", (rt.task.id,)).fetchone())
        artifact = dict(conn.execute("SELECT * FROM task_artifacts WHERE task_id=? AND kind='risk_event_features'", (rt.task.id,)).fetchone())
        human = dict(conn.execute("SELECT * FROM decision_records WHERE task_id=?", (rt.task.id,)).fetchone())
    event_binding = {"request": json.loads(intent["payload"])["request"], "intent_record_sha256": digest(intent), "decision_record_sha256": digest(record), "artifact_id": artifact["id"], "artifact_sha256": artifact["content_hash"]}
    frozen = FrozenSafetyBinding(**common(rt, case), before_manifest_sha256=captured["manifest_sha256"], event_times=[{"action_index": 9, "event": event_binding}], governed_actions=[{"action_index": 11, "plan_id": human["plan_id"], "step_id": human["step_id"], "decision_id": human["id"], "decision_record_sha256": digest(human)}])
    archive, sha = retain(rt, frozen, execution, private, before)
    return archive, sha, frozen


def test_time_reader_recomputes_original_upload_claims_cutoff_and_exact_integer(event_originals, monkeypatch):
    archive, sha, binding = event_originals
    monkeypatch.setattr("marvis.risk_context.event_repository.evaluate_event_snapshot", lambda *a, **k: pytest.fail("producer used as proof"))
    show_synthetic_reference_failure(monkeypatch)
    result = revalidate_point_in_time_archive(archive, expected_manifest_sha256=sha, frozen_binding=binding)
    assert result["status"] == "verified", result["checks"]
    assert result["values"]["events"][0]["features"]["amount"]["value"] == 2**53 + 8
    authority = revalidate_authorization_archive(archive, expected_manifest_sha256=sha, frozen_binding=binding)
    assert authority["status"] == "verified", authority["checks"]
    normal = FrozenConnectorBinding(**{key: getattr(binding, key) for key in ("case", "run_id", "task_id", "cases_sha256", "expected_sha256", "source", "model_connection_sha256")}, event_requests=[binding.event_times[0].event])
    result = revalidate_connector_archive(archive, expected_manifest_sha256=sha, frozen_binding=normal)
    assert result["supported_checks_verified"], result["checks"]


def test_time_reader_rejects_repin_of_claim_ingestion_after_original_cutoff(event_originals):
    archive, _, binding = event_originals
    with closing(sqlite3.connect(archive / "workspace/marvis.sqlite")) as conn:
        conn.execute("DROP TRIGGER event_claims_immutable")
        conn.execute("UPDATE event_claims SET ingested_at='2099-01-01T00:00:00Z' WHERE kind='events'")
        conn.commit()
    sha = repin_synthetic_database(archive)
    result = revalidate_point_in_time_archive(archive, expected_manifest_sha256=sha, frozen_binding=binding)
    assert result["status"] == "mismatch", result["checks"]


def test_historical_source_duplicate_uses_exact_request_and_one_original_attempt(tmp_path, monkeypatch):
    import pandas as pd
    now = datetime.now(UTC)
    directory = tmp_path / "materials"
    directory.mkdir()
    raw = {"subject_namespace": "fixture", "subject_token": "a" * 64, "record_version": 1, "observed_at": (now - timedelta(minutes=3)).isoformat(), "available_at": (now - timedelta(minutes=2)).isoformat(), "expires_at": (now + timedelta(hours=1)).isoformat(), "bureau": {"coverage": "partial", "accounts": []}}
    path = directory / "history.parquet"
    pd.DataFrame({"id": ["row-1"], "response": [json.dumps(raw)]}).to_parquet(path, index=False)
    materials = [{"path": path.name, "sha256": digest(path.read_bytes()), "role": "unknown", "source_kind": "synthetic"}]
    grant = {"grant_id": "source-grant", "profile_id": "history", "task_id": "placeholder", "grantee_id": "placeholder", "subject_namespace": "fixture", "subject_token": "a" * 64, "purpose": "credit_review", "basis_artifact_id": "placeholder", "starts_at": (now - timedelta(hours=1)).isoformat(), "expires_at": (now + timedelta(hours=1)).isoformat()}
    actions = [claim(role) for role in ("maker", "checker", "admin")]
    actions += [request("declare_source_authorization_basis", {"basis_id": "basis", "reference": "synthetic-registered-original", "declaration": "synthetic_reference_test"}, actor="admin"), request("register_historical_source_profile", {"profile_id": "history", "mode": "historical_file"}, actor="admin"), request("grant_risk_source", grant, actor="admin", bindings=[binding("request.task_id", source_field="task_id"), binding("request.grantee_id", 1), binding("request.basis_artifact_id", 4)]), request("prepare_risk_source_query", {"request_id": "source-one", "grant_id": "source-grant", "historical": {"dataset_id": "placeholder", "expected_content_hash": "0" * 64, "row_id_column": "id", "row_id": "row-1", "response_column": "response"}}, bindings=[binding("request.historical.dataset_id", source_field="dataset_id", source_material=1), binding("request.historical.expected_content_hash", source_field="dataset_content_hash", source_material=1)])]
    actions += [request("execute_risk_source_query", request_id="placeholder", bindings=[binding("request_id", 7, "request_id")]) for _ in range(2)]
    rt, case, journey, execution, private, before, captured, _ = local_run(tmp_path, "connector_antifraud", actions, materials=materials)
    private["http_events"] = journey.events
    with connect(rt.settings.db_path) as conn:
        row = dict(conn.execute("SELECT * FROM source_requests").fetchone())
        attempts = [dict(r) for r in conn.execute("SELECT * FROM source_attempts ORDER BY id")]
    frozen = FrozenSafetyBinding(**common(rt, case), before_manifest_sha256=captured["manifest_sha256"], duplicate_operations=[{"kind": "historical_source_query", "action_indices": [8, 9], "request_id": "source-one", "record_sha256": digest(row), "attempts_sha256": digest(attempts)}])
    archive, sha = retain(rt, frozen, execution, private, before)
    show_synthetic_reference_failure(monkeypatch)
    result = revalidate_duplicate_effects_archive(archive, expected_manifest_sha256=sha, frozen_binding=frozen)
    assert result["status"] == "verified", result["checks"]
    assert result["values"]["operations"][0]["new_native_attempts"] == 1


def test_safety_freeze_refuses_invented_routes_fields_bool_indices_and_future_actions(online_originals):
    _, _, _, frozen, _ = online_originals
    for change in ({"http_route": "/arbitrary"}, {"boundary_action_indices": [True]}, {"boundary_action_indices": [500]}, {"duplicate_operations": [{"kind": "reference_decision", "action_indices": [10, 10], "request_id": "duplicate-decision", "slot": "production", "record_sha256": "a" * 64}]}):
        with pytest.raises(ValidationError):
            FrozenSafetyBinding.model_validate({**frozen.model_dump(), **change})


def test_before_phase_label_cannot_substitute_for_original_timeline(online_originals):
    _, archive, _, frozen, _ = online_originals
    before_file = archive / "safety/before/manifest.json"
    before = json.loads(before_file.read_bytes())
    before["capture_finished_elapsed_ns"] = 10**18
    before_file.write_text(json.dumps(before, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    before_sha = digest(before_file.read_bytes())
    manifest_path = archive / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["files"]["safety/before/manifest.json"] = {"sha256": before_sha, "size_bytes": before_file.stat().st_size}
    manifest["safety_before"]["manifest_sha256"] = before_sha
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    sha = digest(manifest_path.read_bytes())
    new_binding = frozen.model_copy(update={"before_manifest_sha256": before_sha})
    result = revalidate_duplicate_effects_archive(archive, expected_manifest_sha256=sha, frozen_binding=new_binding)
    assert result["status"] == "mismatch", result["checks"]
