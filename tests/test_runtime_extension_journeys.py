import time
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from marvis.app import create_app
from marvis.orchestrator.eval.runtime_contracts import RuntimeAction, RuntimeCase
from marvis.orchestrator.eval.runtime_contracts import digest
from marvis.orchestrator.eval.runtime_extensions import bound_input, submit_extension
from marvis.orchestrator.eval.runtime_extensions import start_extension_workflow, extension_step_decision
from marvis.orchestrator.eval.runtime_runner import Journey, _receipts
from marvis.orchestrator.eval.runtime_extension_observations import observe_extensions
from marvis.orchestrator.eval.runtime_scoring import Assertion, _assertion, score_case
from marvis.repositories.tasks import TaskRepository
from marvis.db import StrategyRepository
from marvis.db_schema import connect
from marvis.packs.strategy.dsl import StrategySpec, StrategyRuleSpec, StrategyAction
from marvis.packs.strategy.strategy import build_strategy_from_spec
from tests.test_db import _task_create


def claim(role):
    return RuntimeAction(kind="submit_extension_request", content="确认本次本地参考角色",
                         extension_request={"kind": "claim_extension_role", "actor": role,
                                            "request": {"display_name": "Synthetic " + role, "role": role}})


def test_actual_public_role_claims_are_separate_sessions_and_original_session_survives(tmp_path):
    app = create_app(tmp_path / "workspace")
    case = RuntimeCase(id="extensions", revision="local", family="online_decision", task={"task_type": "strategy"}, business_constraints_source="synthetic local check")
    with TestClient(app) as client:
        client.get("/api/plugins/governance")
        original = dict(client.cookies)
        journey = Journey(client, case, time.monotonic() + 30)
        journey.task_id = "local-task"
        for role in ("maker", "checker", "admin"):
            submit_extension(journey, claim(role), app.state.settings.workspace)
            assert dict(client.cookies) == original
        actors = [output["response"]["id"] for output in journey.extension_outputs]
        assert len(set(actors)) == 3
        assert journey.interventions == 3
        assert all(event["status_code"] == 201 for event in journey.events)
        assert [out["response"]["role"] for out in journey.extension_outputs] == ["maker", "checker", "admin"]


def test_binding_cannot_guess_future_ids_or_inject_routes():
    action = RuntimeAction(kind="submit_extension_request", content="复核已创建的上线提案",
                           extension_request={"kind": "review_production_promotion", "actor": "checker", "promotion_id": "placeholder",
                                              "request": {"decision": "approve", "reason": "合成复核"}},
                           extension_bindings=[{"target": "promotion_id", "source_action": 1, "source_field": "id"}])
    with pytest.raises(ValueError, match="forward_reference"):
        bound_input(action, "task", [])
    assert bound_input(action, "task", [{"kind": "request_production_promotion", "response": {"id": "native-id"}}]).promotion_id == "native-id"
    with pytest.raises(ValueError, match="identity_reference"):
        bound_input(action, "task", [{"kind": "build_decision_package", "response": {"id": "injected"}}])
    with pytest.raises(ValidationError):
        RuntimeAction.model_validate({**action.model_dump(), "extension_bindings": [{"target": "request.url", "source_action": 1, "source_field": "id"}]})


def test_old_cases_keep_exact_original_action_shape_and_digest():
    action = RuntimeAction(kind="message", content="分析")
    assert action.model_dump() == {"kind": "message", "content": "分析", "tool": ""}
    with pytest.raises(ValidationError):
        RuntimeAction(kind="message", content="分析", extension_request=claim("maker").extension_request)


def native_journey(tmp_path, family, *, task_run_mode="manual", **options):
    app = create_app(tmp_path / "workspace", **options)
    task = TaskRepository(app.state.settings.db_path).create_task(_task_create(task_type="strategy", run_mode=task_run_mode))
    case = RuntimeCase(id="extensions", revision="synthetic-local", family=family, task={"task_type": "strategy"}, business_constraints_source="synthetic public requests; no hidden acceptance answers")
    client = TestClient(app)
    journey = Journey(client, case, time.monotonic() + 120)
    journey.task_id = task.id
    for role in ("maker", "checker", "admin"):
        action = claim(role)
        submit_extension(journey, action, app.state.settings.workspace)
        journey.declared_test_actions = getattr(journey, "declared_test_actions", []) + [action]
    return app, journey


def binding(target, source_action=None, source_field="id", **source):
    return {"target": target, "source_action": source_action, "source_field": source_field, **source}


def submit(app, journey, kind, request=None, *, actor="maker", bindings=(), **fields):
    item = {"kind": kind, "actor": actor, **fields}
    if request is not None:
        item["request"] = request
    action = RuntimeAction(kind="submit_extension_request", content="确认本次合成业务输入", extension_request=item, extension_bindings=list(bindings))
    submit_extension(journey, action, app.state.settings.workspace)
    journey.declared_test_actions.append(action)
    return journey.extension_outputs[-1]["response"]


def observed(app, journey):
    evidence, private = _receipts(app.state.settings.workspace, journey.task_id, case=journey.case, journey=journey)
    observation, values = observe_extensions(app.state.settings.workspace, journey.task_id, journey.extension_outputs, execution=evidence, bound_outputs=private["outputs"], decisions=getattr(journey, "extension_decisions", ()), http_events=journey.events)
    evidence["extension_observation"] = observation
    private["extension_values"] = values
    return evidence, private


def workflow(journey, tool, proposal_action):
    start = RuntimeAction(kind="start_extension_workflow", content="启动原生提案确认旅程", tool=tool, extension_source_action=proposal_action)
    decision = RuntimeAction(kind="approve_step", tool=tool, content="已核对本次明确来源、时间及本地执行边界")
    start_extension_workflow(journey, start)
    assert extension_step_decision(journey, decision)
    journey.declared_test_actions.extend((start, decision))


def test_online_journey_uses_real_three_role_chain_native_package_and_decision(tmp_path):
    app, journey = native_journey(tmp_path, "online_decision")
    spec = StrategySpec(strategy_type="approval", rules=(StrategyRuleSpec(rule_id="low", priority=1, condition={"op": "compare", "field": "x1", "operator": "<", "value": .5}, action=StrategyAction(type="approval", reason_code="LOW")),), default_action=StrategyAction(type="review", reason_code="REVIEW"))
    strategy = replace(build_strategy_from_spec(spec, score_col="x1", description="synthetic validated prerequisite"), id="journey-rules")
    StrategyRepository(app.state.settings.db_path).create_strategy(journey.task_id, strategy)
    # This synthetic prerequisite is independent of the production extension
    # under test; it does not claim strategy development acceptance.
    with connect(app.state.settings.db_path) as conn:
        conn.execute("UPDATE strategies SET asset_status='validated' WHERE id=?", (strategy.id,))
    submit(app, journey, "build_decision_package", {"package_kind": "rule_only", "strategy_id": strategy.id, "strategy_version": 1, "decision_node": "underwriting", "raw_schema": [{"name": "x1", "type": "number"}]})
    submit(app, journey, "request_production_promotion", {"environment": "local-reference", "deployment_slot": "production", "strategy_id": strategy.id, "strategy_version": 1, "decision_package_hash": "0" * 64, "reason": "synthetic exact package"}, bindings=[binding("request.decision_package_hash", 4, "package_hash")])
    for role in ("checker", "admin"):
        submit(app, journey, "review_production_promotion", {"decision": "approve", "reason": "independent synthetic review"}, actor=role, promotion_id="placeholder", bindings=[binding("promotion_id", 5)])
    submit(app, journey, "install_decision_package", {"promotion_id": "placeholder", "probe_features": {"x1": .1}}, actor="admin", bindings=[binding("request.promotion_id", 5)])
    submit(app, journey, "activate_production_promotion", {"verifier_id": "local-reference.v1", "receipt_id": "placeholder", "reason": "observed local package"}, actor="admin", promotion_id="placeholder", bindings=[binding("promotion_id", 5), binding("request.receipt_id", 8, "activation_evidence.receipt_id")])
    req = {"request_id": "journey-one", "decision_node": "underwriting", "expected_package_hash": "0" * 64, "features": {"x1": .1}}
    first = submit(app, journey, "submit_reference_decision", req, bindings=[binding("request.expected_package_hash", 4, "package_hash")])
    assert submit(app, journey, "replay_reference_decision", req, bindings=[binding("request.expected_package_hash", 4, "package_hash")]) == first
    evidence, private = observed(app, journey)
    assert private["extension_values"]["decisions"] == [{"request_id": "journey-one", "action_type": "approval", "matched_rule_id": "low"}]
    check = Assertion(kind="extension_value", path=["decisions", 0, "action_type"], value="approval")
    assert _assertion(check, {"execution": evidence}, private)
    private["extension_values"]["decisions"][0]["action_type"] = "review"
    assert not _assertion(check, {"execution": evidence}, private)
    assert len({out["response"]["id"] for out in journey.extension_outputs[:3]}) == 3


def material_refs(*targets):
    return [binding(target + "_id", 4, "source_artifact_id") for target in targets] + [binding(target + "_hash", 4, "source_artifact_hash") for target in targets]


def test_collection_journey_actual_cashflow_and_two_governed_effects(tmp_path):
    from tests.test_collection_cashflows import UNIT, flow, request
    from tests.test_collection_planning import AT, case, history, policy, spec
    app, journey = native_journey(tmp_path, "collections")
    submit(app, journey, "declare_collection_material", {"material_id": "history", "description": "synthetic local declarations", "declarations": {"synthetic": True}})
    refs = {"source_artifact_id": "placeholder", "source_artifact_hash": "0" * 64}
    submit(app, journey, "create_collection_case", {"case_id": "case-1", "subject_namespace": "fixture", "subject_token": "a" * 64, "unit": UNIT, "opening_balance_minor": 10000, "opened_at": "2026-08-01T00:00:00Z", "source_assurance": "historical_import_unverified", **refs}, bindings=material_refs("request.source_artifact"))
    from marvis.collection.actions import CollectionPolicy
    p = policy(unit=UNIT, basis_artifact_id="placeholder", basis_artifact_hash="0" * 64, max_estimated_active_cost_minor=100)
    declared_policy = p.model_dump()
    declared_policy["queues"][0]["max_active_actions"] = 5
    p = CollectionPolicy.model_validate(declared_policy)
    batch = {"batch_id": "journey-batch", "execution_mode": "local_reference", "strategy": spec(p).to_dict(), "policy": p.model_dump(), "cases": [case("case-1", subject_namespace="fixture", **refs).model_dump()], "histories": [history(subject_namespace="fixture", **refs).model_dump()], "as_of": AT, "knowledge_cutoff": AT}
    submit(app, journey, "prepare_collection_batch", batch, bindings=[*material_refs("request.policy.basis_artifact", "request.cases[].source_artifact", "request.histories[].source_artifact"), binding("request.strategy.default_action.value.policy_hash", source_field="resolved_policy_hash")])
    with pytest.raises(ValueError, match="not_executed"):
        observed(app, journey)
    workflow(journey, "collection.queue_batch", 6)
    workflow(journey, "collection.execute_reference", 6)
    submit(app, journey, "read_collection_effects", batch_id="placeholder", bindings=[binding("batch_id", 6, "batch_id")])
    submit(app, journey, "import_collection_cashflows", {"events": [flow(refs).model_dump(), flow(refs, "reversal", kind="payment_reversal", amount_minor=1000, reverses={"source_id": "bank", "event_id": "payment-1"}).model_dump()]}, bindings=material_refs("request.events[].source_artifact"))
    submit(app, journey, "reconcile_collection_cashflows", request(refs).model_dump(), bindings=material_refs("request.coverage[].source_artifact"))
    evidence, private = observed(app, journey)
    result = private["extension_values"]
    assert result["batches"][0]["status"] == "completed"
    assert result["batches"][0]["customer_contacted"] is False
    assert result["reconciliations"][0]["amounts"]["net_payments_minor"] == 2000
    assert sum(step["binding_verified"] for step in evidence["steps"]) == 2
    with connect(app.state.settings.db_path) as conn:
        actors = [row[0] for row in conn.execute("SELECT principal_id FROM decision_records WHERE decision='approve'")]
    assert actors == [journey.extension_outputs[1]["response"]["id"]] * 2
    # Run the same public, predeclared complete inputs through Journey.run on a
    # fresh product workspace. No expected answer drives an HTTP action.
    frozen = RuntimeCase.model_validate({**journey.case.model_dump(), "actions": [a.model_dump() for a in journey.declared_test_actions]})
    before = digest(frozen.model_dump())
    other = create_app(tmp_path / "second" / "workspace")
    with TestClient(other) as client:
        repeat = Journey(client, frozen, time.monotonic() + frozen.budget.wall_seconds)
        repeat.run(tmp_path, other.state.settings.workspace)
        _, actual = observed(other, repeat)
    assert actual["extension_values"]["batches"][0]["status"] == "completed"
    assert repeat.count <= frozen.budget.max_http_requests
    assert digest(frozen.model_dump()) == before


def test_connector_event_journey_requires_plan_and_real_worker_before_observation(tmp_path, monkeypatch):
    import pandas as pd
    from tests.test_event_features import source, event, coverage, query
    app, journey = native_journey(tmp_path, "connector_antifraud", task_run_mode="agent")
    submit(app, journey, "declare_source_authorization_basis", {"basis_id": "event-basis", "reference": "synthetic-event-declarations", "declaration": "synthetic_reference_test"}, actor="admin")
    src = source(task_id="placeholder")
    submit(app, journey, "register_risk_event_source", src.model_dump(), actor="admin", bindings=[binding("request.task_id", source_field="task_id")])
    now = datetime.now(UTC)
    grant = {"grant_id": "event-grant", "task_id": "placeholder", "source_id": src.source_id, "source_contract_hash": "0" * 64, "grantee_id": "placeholder", "permissions": ["read", "write"], "purpose": "synthetic event declarations", "basis_artifact_id": "placeholder", "starts_at": (now - timedelta(hours=1)).isoformat(), "expires_at": (now + timedelta(hours=1)).isoformat()}
    submit(app, journey, "grant_risk_event_source", grant, actor="admin", bindings=[binding("request.task_id", source_field="task_id"), binding("request.grantee_id", 1), binding("request.basis_artifact_id", 4), binding("request.source_contract_hash", 5, "contract_hash")])
    for index, (kind, claims) in enumerate((("events", [event(values={"amount_minor": 2**53 + 1}), event("second", values={"amount_minor": 7})]), ("coverage", [coverage()])), 1):
        path = tmp_path / (kind + ".parquet")
        pd.DataFrame({"claim": [json.dumps(row.model_dump()) for row in claims]}).to_parquet(path, index=False)
        with path.open("rb") as stream:
            uploaded = journey.json_request("POST", f"/api/tasks/{journey.task_id}/datasets/upload", label="upload_material", files={"file": (path.name, stream)}, data={"role": "unknown"})
        journey.uploaded_materials.append({"datasets": uploaded["datasets"]})
        item = {"grant_id": "event-grant", "dataset": {"source_id": src.source_id, "source_contract_hash": "0" * 64, "dataset_id": "placeholder", "expected_content_hash": "0" * 64, "kind": kind, "json_column": "claim"}}
        submit(app, journey, "import_risk_event_dataset", item, bindings=[binding("request.dataset.source_contract_hash", 5, "contract_hash"), binding("request.dataset.dataset_id", source_field="dataset_id", source_material=index), binding("request.dataset.expected_content_hash", source_field="dataset_content_hash", source_material=index)])
    contract = query(src, availability_mode="retrospective_declared").model_dump()
    submit(app, journey, "prepare_risk_event_features", {"request_id": "event-one", "grant_id": "event-grant", "contract": contract}, bindings=[binding("request.contract.source_contract_hash", 5, "contract_hash"), binding("request.contract.knowledge_cutoff", 8, "ingested_at")])
    with pytest.raises(ValueError):
        observed(app, journey)
    workflow(journey, "risk_context.replay_events", 9)
    assert any(item["stage"] == "human_extension_agent_plan_start" and item["status_code"] == 202 for item in journey.events)
    submit(app, journey, "read_risk_event_evidence", request_id="placeholder", grant_id="event-grant", bindings=[binding("request_id", 9, "request_id")])
    monkeypatch.setattr("marvis.risk_context.event_repository.evaluate_event_snapshot", lambda *a, **k: pytest.fail("producer algorithm used as proof"))
    evidence, private = observed(app, journey)
    value = private["extension_values"]["events"][0]
    assert value["features"]["amount"]["value"] == 2**53 + 8
    assert value["features"]["transactions"]["value"] == 2
    assert evidence["steps"][0]["binding_verified"]
    assert all(event["status_code"] < 400 for event in journey.events)
    with pytest.raises(ValueError, match="governed_native_producer"):
        observe_extensions(app.state.settings.workspace, journey.task_id, journey.extension_outputs, execution={"steps": []}, bound_outputs={})
    monkeypatch.undo()
    materials = [{"path": kind + ".parquet", "sha256": digest((tmp_path / (kind + ".parquet")).read_bytes()), "role": "unknown", "source_kind": "synthetic"} for kind in ("events", "coverage")]
    frozen = RuntimeCase.model_validate({**journey.case.model_dump(), "materials": materials, "actions": [action.model_dump() for action in journey.declared_test_actions]})
    identity = digest(frozen.model_dump())
    other = create_app(tmp_path / "second" / "workspace")
    with TestClient(other) as client:
        repeat = Journey(client, frozen, time.monotonic() + frozen.budget.wall_seconds)
        repeat.run(tmp_path, other.state.settings.workspace)
        _, actual = observed(other, repeat)
    assert actual["extension_values"]["events"][0]["features"]["amount"]["value"] == 2**53 + 8
    assert repeat.count <= frozen.budget.max_http_requests
    assert digest(frozen.model_dump()) == identity


def test_connector_historical_source_full_http_query_keeps_partial_unknown(tmp_path):
    import pandas as pd
    from tests.test_risk_source_runtime import source_record
    app, journey = native_journey(tmp_path, "connector_antifraud")
    submit(app, journey, "declare_source_authorization_basis", {"basis_id": "source-basis", "reference": "synthetic-history", "declaration": "synthetic_reference_test"}, actor="admin")
    submit(app, journey, "register_historical_source_profile", {"profile_id": "history", "mode": "historical_file"}, actor="admin")
    now = datetime.now(UTC)
    grant = {"grant_id": "source-grant", "profile_id": "history", "task_id": "placeholder", "grantee_id": "placeholder", "subject_namespace": "reference.people", "subject_token": "a" * 64, "purpose": "credit_review", "basis_artifact_id": "placeholder", "starts_at": (now - timedelta(hours=1)).isoformat(), "expires_at": (now + timedelta(hours=1)).isoformat()}
    submit(app, journey, "grant_risk_source", grant, actor="admin", bindings=[binding("request.task_id", source_field="task_id"), binding("request.grantee_id", 1), binding("request.basis_artifact_id", 4)])
    path = tmp_path / "history.parquet"
    pd.DataFrame({"row": ["one"], "response": [json.dumps(source_record())]}).to_parquet(path, index=False)
    with path.open("rb") as stream:
        uploaded = journey.json_request("POST", f"/api/tasks/{journey.task_id}/datasets/upload", label="upload_material", files={"file": (path.name, stream)}, data={"role": "unknown"})
    journey.uploaded_materials.append({"datasets": uploaded["datasets"]})
    req = {"request_id": "history-one", "grant_id": "source-grant", "historical": {"dataset_id": "placeholder", "expected_content_hash": "0" * 64, "row_id_column": "row", "row_id": "one", "response_column": "response"}}
    submit(app, journey, "prepare_risk_source_query", req, bindings=[binding("request.historical.dataset_id", source_field="dataset_id", source_material=1), binding("request.historical.expected_content_hash", source_field="dataset_content_hash", source_material=1)])
    submit(app, journey, "execute_risk_source_query", request_id="placeholder", bindings=[binding("request_id", 7, "request_id")])
    submit(app, journey, "read_risk_source_evidence", request_id="placeholder", bindings=[binding("request_id", 7, "request_id")])
    evidence, private = observed(app, journey)
    values = private["extension_values"]["sources"][0]
    assert values["account_count"] is None and values["totals_by_currency"] is None
    assert values["automated_clearance"] is False
    assert _assertion(Assertion(kind="extension_value", path=["sources", 0, "account_count"], value=None), {"execution": evidence}, private)


def test_binding_rejects_cross_semantic_ids_and_duplicate_unknown_native_sources():
    action = RuntimeAction(kind="submit_extension_request", content="review", extension_request={"kind": "review_production_promotion", "actor": "checker", "promotion_id": "placeholder", "request": {"decision": "approve", "reason": "synthetic review"}}, extension_bindings=[binding("promotion_id", 1)])
    with pytest.raises(ValueError, match="identity_reference"):
        bound_input(action, "task", [{"kind": "claim_extension_role", "actor": "maker", "response": {"id": "principal"}}])
    with pytest.raises(ValidationError):
        RuntimeAction(kind="start_extension_workflow", tool="arbitrary.execute", content="start", extension_source_action=1)
    with pytest.raises(ValidationError):
        RuntimeAction(kind="message", content="hello", extension_source_action=1)


def test_operations_extension_actual_calendar_run_and_outcome_carrier(tmp_path):
    from marvis.operations import MonitoringOutcome
    fixed = datetime(2026, 10, 1, 10, tzinfo=UTC)
    app, journey = native_journey(tmp_path, "online_decision", operations_clock=lambda: fixed,
                                 operations_executor_allowlist={"synthetic-local": lambda req: MonitoringOutcome(level="green", evidence_ref="synthetic-period", evidence_hash=digest({"period": req.period.key}))})
    payload = {"expected_revision": 0, "schedule": {"schedule_id": "synthetic-operations", "revision": 1, "monitoring_ref": "synthetic-local", "active_from": (fixed - timedelta(minutes=2)).isoformat(), "calendar": {"anchor_at": (fixed - timedelta(minutes=2)).isoformat(), "interval_seconds": 60}, "catch_up_budget": 1, "lease_seconds": 30, "retry_policy": {"max_attempts": 2, "initial_backoff_seconds": 0., "multiplier": 1., "max_backoff_seconds": 0.}, "enabled": True}}
    submit(app, journey, "publish_operations_schedule", payload)
    submit(app, journey, "tick_operations", {"catch_up_budget": 1, "notification_budget": 1}, actor="admin")
    _, private = observed(app, journey)
    assert private["extension_values"]["schedules"] == [{"period_count": 1, "run_count": 1, "outcome_count": 1}]
    with connect(app.state.settings.db_path) as conn:
        contract = json.loads(conn.execute("SELECT contract_json FROM operations_schedules").fetchone()[0])
        assert contract["execution_actor_id"] == journey.extension_outputs[0]["response"]["id"]
        contract["execution_actor_id"] = journey.extension_outputs[2]["response"]["id"]
        # Simulate a compromised archive after bypassing the native immutability
        # guard; recomputing a public hash must not bless a different publisher.
        conn.execute("DROP TRIGGER operations_schedules_no_update")
        conn.execute("UPDATE operations_schedules SET contract_json=?,contract_hash=?", (json.dumps(contract), digest(contract)))
    with pytest.raises(ValueError, match="schedule_contract_mismatch"):
        observed(app, journey)


def test_scoring_never_accepts_unbound_response_flag_or_bool_count():
    check = Assertion(kind="extension_value", path=["events", 0, "population_count"], value=1)
    values = {"events": [{"population_count": True}]}
    record = {"execution": {"extension_observation": {"verified": True, "value_sha256": digest(values)}}}
    assert not _assertion(check, record, {"extension_values": values})
    record["execution"]["extension_observation"]["value_sha256"] = "0" * 64
    assert not _assertion(check, record, {"extension_values": {"events": [{"population_count": 1}]}})


def _score_observation(journey, evidence, private, result, path, value):
    expected = json.dumps({"cases": {journey.case.id: {"result": result, "assertions": [{"kind": "extension_value", "path": path, "value": value}]}}}).encode()
    record = {"runtime_status": "completed", "execution": evidence, "http_events": journey.events, "llm_events": [], "human_interventions": journey.interventions}
    return score_case(journey.case, record, private, expected, model_source="fixture_model")


def test_actual_public_http_rejection_is_observed_without_error_prose_or_success(tmp_path):
    app, journey = native_journey(tmp_path, "connector_antifraud")
    submit(app, journey, "read_risk_source_evidence", request_id="no-such-request")
    output = journey.extension_outputs[-1]
    assert output["response"] == {"status": "http_rejected", "http_status": 404}
    evidence, private = observed(app, journey)
    assert evidence["extension_observation"]["measurement_count"] == 0
    assert evidence["extension_observation"]["rejection_count"] == 1
    path = ["rejections", 0, "status"]
    assert _score_observation(journey, evidence, private, "rejected", path, "http_rejected")["passed"]
    assert not _score_observation(journey, evidence, private, "done", path, "http_rejected")["passed"]
    with pytest.raises(ValueError, match="http_fact_unavailable"):
        observe_extensions(app.state.settings.workspace, journey.task_id, journey.extension_outputs)


def test_explicit_native_plan_rejection_never_becomes_batch_measurement(tmp_path):
    from tests.test_collection_planning import AT, case, history, policy, spec
    app, journey = native_journey(tmp_path, "collections")
    submit(app, journey, "declare_collection_material", {"material_id": "history", "description": "synthetic local declarations", "declarations": {"synthetic": True}})
    refs = {"source_artifact_id": "placeholder", "source_artifact_hash": "0" * 64}
    p = policy(basis_artifact_id="placeholder", basis_artifact_hash="0" * 64, max_estimated_active_cost_minor=100)
    from marvis.collection.actions import CollectionPolicy
    declared = p.model_dump()
    declared["queues"][0]["max_active_actions"] = 5
    p = CollectionPolicy.model_validate(declared)
    case_input = case("case-1", **refs).model_dump()
    submit(app, journey, "create_collection_case", {"case_id": "case-1", "subject_namespace": case_input["subject_namespace"], "subject_token": case_input["subject_token"], "unit": p.unit.model_dump(), "opening_balance_minor": 10000, "opened_at": "2026-08-01T00:00:00Z", "source_assurance": "historical_import_unverified", **refs}, bindings=material_refs("request.source_artifact"))
    batch = {"batch_id": "rejected-batch", "execution_mode": "local_reference", "strategy": spec(p).to_dict(), "policy": p.model_dump(), "cases": [case_input], "histories": [history(**refs).model_dump()], "as_of": AT, "knowledge_cutoff": AT}
    submit(app, journey, "prepare_collection_batch", batch, bindings=[*material_refs("request.policy.basis_artifact", "request.cases[].source_artifact", "request.histories[].source_artifact"), binding("request.strategy.default_action.value.policy_hash", source_field="resolved_policy_hash")])
    assert journey.extension_outputs[-1]["http_status"] == 201
    start_extension_workflow(journey, RuntimeAction(kind="start_extension_workflow", content="准备原生提案复核", tool="collection.queue_batch", extension_source_action=6))
    assert extension_step_decision(journey, RuntimeAction(kind="reject_step", content="明确拒绝本次提案，不执行", tool="collection.queue_batch"))
    evidence, private = observed(app, journey)
    assert private["extension_values"]["batches"] == []
    assert private["extension_values"]["rejections"][0]["status"] == "rejected"
    assert evidence["extension_observation"]["measurement_count"] == 0
    assert evidence["plans"][0]["status"] == "cancelled"
    path = ["rejections", 0, "status"]
    assert _score_observation(journey, evidence, private, "rejected", path, "rejected")["passed"]
    assert not _score_observation(journey, evidence, private, "done", path, "rejected")["passed"]
    with connect(app.state.settings.db_path) as conn:
        assert conn.execute("SELECT count(*) FROM plan_step_runs").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM effect_executions").fetchone()[0] == 0
        conn.execute("DROP TRIGGER trg_decision_records_immutable_update")
        conn.execute("UPDATE decision_records SET input_hash=?", ("0" * 64,))
    with pytest.raises(ValueError, match="rejection_binding_mismatch"):
        observed(app, journey)
