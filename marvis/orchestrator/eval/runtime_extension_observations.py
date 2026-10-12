"""Local native observations; independent archive acceptance remains separate.

Read authenticated registered carriers after runtime shutdown and recompute the
supported business values. HTTP responses alone never establish a measurement.
"""
import json
from pathlib import Path

from marvis.db_schema import connect
from marvis.reference_decision.packages import PackageStore
from marvis.risk_context.event_service import EventService
from marvis.risk_context.event_contracts import canonical_json as event_canonical
from marvis.risk_context.source_service import SourceService
from marvis.collection.ledger import CollectionLedger
from marvis.collection.service import CollectionService
from marvis.settings import Settings

from .runtime_contracts import digest
from .runtime_extensions_archive import measured_values
from .runtime_online_reference import rule_decision_reference, calendar_reference, schedule_contract_reference
from .runtime_connector_reference import event_features_reference, source_assessment_reference
from .runtime_collections_reference import reconciliation_reference, batch_preview_reference
from .runtime_safety_reference import governance_input_reference


def _operations(conn, request, publisher):
    declared = request["schedule"]
    row = conn.execute("SELECT * FROM operations_schedules WHERE schedule_id=? AND revision=?", (declared["schedule_id"], declared["revision"])).fetchone()
    if row is None:
        raise ValueError("extension_native_schedule_unavailable")
    contract = json.loads(row["contract_json"])
    principal = conn.execute("SELECT role,status FROM production_principals WHERE local_principal_id=?", (publisher,)).fetchone()
    if principal is None or principal["role"] not in {"maker", "admin"} or principal["status"] != "active":
        raise ValueError("extension_native_schedule_publisher_unavailable")
    expected = schedule_contract_reference(declared, execution_actor_id=publisher)
    if contract != expected or digest(contract) != row["contract_hash"]:
        raise ValueError("extension_native_schedule_contract_mismatch")
    tables = {}
    for table in ("operations_periods", "operations_runs", "operations_outcomes", "operations_notification_outbox"):
        rows = conn.execute(f"SELECT * FROM {table} WHERE schedule_id=?", (declared["schedule_id"],)).fetchmany(10001)  # nosec B608 -- table is from the fixed four-table operations tuple; declared schedule id is bound
        if len(rows) > 10000:
            raise ValueError("extension_native_operations_budget")
        tables[table] = [dict(row) for row in rows]
    for period in tables["operations_periods"]:
        if period["schedule_revision"] != declared["revision"]:
            raise ValueError("extension_native_mixed_schedule_revisions")
        calendar_reference(contract, period)
        runs = [row for row in tables["operations_runs"] if row["period_key"] == period["period_key"]]
        if sorted(row["attempt_number"] for row in runs) != list(range(1, period["attempt_count"] + 1)):
            raise ValueError("extension_native_operations_run_sequence")
        outcomes = [row for row in tables["operations_outcomes"] if row["period_key"] == period["period_key"]]
        if period["state"] == "succeeded" and (len(outcomes) != 1 or outcomes[0]["outcome_id"] != period["outcome_id"] or outcomes[0]["run_id"] not in {row["run_id"] for row in runs}):
            raise ValueError("extension_native_operations_outcome_closure")
    for row in tables["operations_outcomes"]:
        if digest(json.loads(row["outcome_json"])) != row["outcome_hash"]:
            raise ValueError("extension_native_operations_outcome_hash")
    for row in tables["operations_notification_outbox"]:
        if digest(json.loads(row["payload_json"])) != row["payload_hash"] or row["outcome_id"] not in {item["outcome_id"] for item in tables["operations_outcomes"]}:
            raise ValueError("extension_native_notification_outcome_mismatch")
    return {"period_count": len(tables["operations_periods"]), "run_count": len(tables["operations_runs"]), "outcome_count": len(tables["operations_outcomes"])}


def _producer(execution, outputs, tool, artifact_id):
    matches = [step for step in execution.get("steps", ()) if step["tool"] == tool
               and step["status"] == "done" and step.get("binding_verified") is True
               and outputs.get(step["id"], {}).get("artifact_id") == artifact_id
               and any(run.get("invocation_id") == step.get("producer_invocation_id") and run["status"] == "succeeded"
                       for run in step.get("runs", ()))]
    if len(matches) != 1:
        raise ValueError("extension_governed_native_producer_unavailable")


def _http_fact(events, ordinal, stage, status):
    matches = [event for event in events if event.get("ordinal") == ordinal]
    if (type(ordinal) is not int or len(matches) != 1 or matches[0].get("stage") != stage
            or matches[0].get("status_code") != status):
        raise ValueError("extension_rejection_http_fact_unavailable")


def _rejected_workflows(conn, task_id, proposal, expected_inputs, tools, outputs, decisions, events):
    result = []
    for item in decisions:
        if item["request"]["decision"] != "reject" or item["tool"] not in tools:
            continue
        if type(item.get("proposal_action")) is not int or not 1 <= item["proposal_action"] <= len(outputs):
            raise ValueError("extension_rejected_proposal_unbound")
        if outputs[item["proposal_action"] - 1] is not proposal:
            continue
        _http_fact(events, item["http_ordinal"], "human_extension_rejection", 202)
        response = item["response"]
        decision = conn.execute("SELECT * FROM decision_records WHERE id=? AND task_id=?", (response["decision_id"], task_id)).fetchone()
        step = conn.execute("SELECT s.*,p.status AS plan_status,p.replan_count FROM plan_steps s JOIN plans p ON p.id=s.plan_id WHERE s.id=? AND p.task_id=?", (response["step_id"], task_id)).fetchone()
        claims = {output["response"]["id"] for output in outputs if output["kind"] == "claim_extension_role" and output["actor"] == item["actor"] and output.get("http_status", 201) < 400}
        if (decision is None or step is None or len(claims) != 1 or decision["principal_id"] not in claims
                or decision["principal_id"] != response["principal_id"] or response["decision"] != "reject" or response["approval_id"] is not None
                or step["plan_id"] != response["plan_id"] or decision["plan_id"] != step["plan_id"] or decision["step_id"] != step["id"]
                or step["tool_plugin"] + "." + step["tool_name"] != item["tool"] or not decision["tool_ref"].startswith(item["tool"] + "@")
                or decision["plan_revision"] != step["replan_count"] or decision["decision"] != "reject" or decision["reason"] != item["request"]["reason"]
                or json.loads(step["inputs_json"]) != expected_inputs or decision["input_hash"] != governance_input_reference(expected_inputs)
                or step["status"] != "skipped" or step["plan_status"] != "cancelled" or step["confirmed"] or step["output_ref"]
                or step["error"] != "human_rejected:" + decision["id"]):
            raise ValueError("extension_native_rejection_binding_mismatch")
        if conn.execute("SELECT 1 FROM plan_step_runs WHERE step_id=?", (step["id"],)).fetchone() is not None:
            raise ValueError("extension_rejected_step_has_producer_activity")
        if conn.execute("SELECT 1 FROM effect_executions e JOIN approval_records a ON a.id=e.approval_id WHERE a.step_id=?", (step["id"],)).fetchone() is not None:
            raise ValueError("extension_rejected_step_has_effect_activity")
        result.append({"kind": proposal["kind"], "status": "rejected", "tool": item["tool"], "decision_id": decision["id"], "http_status": 202})
    return result


def observe_extensions(workspace, task_id, outputs, *, execution=None, bound_outputs=None, decisions=(), http_events=()):
    settings = Settings(Path(workspace))
    values = {"decisions": [], "sources": [], "events": [], "reconciliations": [], "batches": [], "schedules": [], "rejections": []}
    identities, seen = [], set()
    with connect(settings.db_path) as conn:
        for output in outputs:
            kind, declaration = output["kind"], output["request"]
            status = output.get("http_status", 200)
            if status >= 400:
                if status not in {400, 401, 403, 404, 409, 422} or output["response"] != {"status": "http_rejected", "http_status": status}:
                    raise ValueError("extension_http_rejection_invalid")
                _http_fact(http_events, output["http_ordinal"], "human_extension_" + kind, status)
                values["rejections"].append({"kind": kind, "status": "http_rejected", "http_status": status, "request_sha256": digest(declaration)})
                continue
            request = declaration.get("request", {})
            if kind in {"submit_reference_decision", "replay_reference_decision"}:
                identity = ("decisions", declaration["slot"], request["request_id"])
                if identity in seen:
                    continue
                row = conn.execute("SELECT * FROM reference_decisions WHERE environment=? AND request_id=?", ("local-reference:" + declaration["slot"], request["request_id"])).fetchone()
                if row is None or row["status"] != "completed":
                    raise ValueError("extension_native_decision_unavailable")
                effective = {key: value for key, value in request.items() if key != "event_evidence" or value is not None}
                if row["input_hash"] != digest(effective) or row["package_hash"] != request["expected_package_hash"]:
                    raise ValueError("extension_native_decision_binding_mismatch")
                manifest = PackageStore(settings, settings.plugin_admin_token_path.read_text().strip().encode()).get(row["package_hash"])
                strategy = conn.execute("SELECT task_id FROM strategies WHERE id=?", (manifest["configuration"]["strategy_id"],)).fetchone()
                if strategy is None or strategy["task_id"] != task_id:
                    raise ValueError("extension_native_decision_task_mismatch")
                actual = json.loads(row["response_json"])
                if actual != output["response"] or actual["status"] != "decided":
                    raise ValueError("extension_native_decision_result_mismatch")
                expected = rule_decision_reference(manifest, effective)
                measured_values(actual, expected)
                values["decisions"].append({"request_id": request["request_id"], "action_type": expected["action"]["type"], "matched_rule_id": expected["matched_rule_id"]})
            elif kind == "prepare_risk_source_query":
                identity = ("sources", request["request_id"])
                row = conn.execute("SELECT * FROM source_requests WHERE task_id=? AND id=?", (task_id, request["request_id"])).fetchone()
                if row is None or row["state"] != "completed" or json.loads(row["contract"])["query"] != request:
                    raise ValueError("extension_native_source_not_completed")
                evidence = SourceService(settings).evidence(task_id, request["request_id"], row["actor_id"])
                receipt = evidence["receipt"]
                if evidence["summary"]["status"] == "unauthorized":
                    raise ValueError("extension_native_source_measurement_unavailable")
                if evidence["normalized_source"] is None:
                    expected = {"status": "no_record", "missing_reasons": ["provider:no_record"], "finding_codes": [], "account_count": None, "inquiry_count": None, "totals_by_currency": None}
                else:
                    expected = source_assessment_reference(evidence["normalized_source"], receipt["recorded_at"])
                measured_values(receipt, {"recorded_status": expected["status"], "missing_reasons": expected["missing_reasons"], "finding_codes": expected["finding_codes"]})
                values["sources"].append({"request_id": request["request_id"], **{key: expected[key] for key in ("status", "account_count", "inquiry_count", "totals_by_currency", "missing_reasons", "finding_codes")}, "automated_clearance": False})
            elif kind == "prepare_risk_event_features":
                identity = ("events", request["request_id"])
                service = EventService(settings)
                row = conn.execute("SELECT * FROM event_requests WHERE task_id=? AND request_id=?", (task_id, request["request_id"])).fetchone()
                if row is None:
                    raise ValueError("extension_native_event_intent_unavailable")
                intent = service.repo._body(row)
                if intent["request"] != request:
                    raise ValueError("extension_native_event_intent_mismatch")
                expected_inputs = {"request_id": request["request_id"], "proposal_hash": digest(intent), "contract": request["contract"]}
                rejected = _rejected_workflows(conn, task_id, output, expected_inputs, {"risk_context.replay_events"}, outputs, decisions, http_events)
                values["rejections"].extend(rejected)
                service.repo._access(conn, task_id, request["contract"]["source_id"], request["grant_id"], intent["actor_id"], "read")
                decision_id = "request:" + digest(intent)
                decision = conn.execute("SELECT * FROM event_decisions WHERE task_id=? AND decision_id=?", (task_id, decision_id)).fetchone()
                if decision is None:
                    if rejected:
                        continue
                    raise ValueError("extension_native_event_not_executed")
                receipt = service.repo._decision_body(decision, task_id, decision_id)
                path = service._path(intent)
                artifact = service.artifacts.get_for_task_kind_path(task_id, "risk_event_features", str(path))
                payload = {"intent": intent, "receipt": receipt}
                envelope = {"body": payload, "signature": service.repo._sign(payload)}
                if artifact is None or artifact["origin_tool"] != "risk_context.replay_events.v1" or artifact["content_hash"] != digest(envelope):
                    raise ValueError("extension_native_event_artifact_mismatch")
                service._check_file(path, event_canonical(envelope).encode())
                expected = event_features_reference(receipt["snapshot"])
                measured_values(receipt["result"], expected)
                _producer(execution or {}, bound_outputs or {}, "risk_context.replay_events", artifact["id"])
                values["events"].append({"request_id": request["request_id"], "status": expected["status"], "population_count": expected["population_count"], "features": {key: {field: value[field] for field in ("status", "value", "unit", "observed_member_count", "missing_reasons")} for key, value in expected["features"].items()}, "automated_clearance": False})
                # A signed event receipt also requires the real governed producer
                # closure captured by _receipts, verified by the caller below.
                identities.append({"kind": kind, "artifact_id": artifact["id"], "content_hash": artifact["content_hash"]})
            elif kind == "reconcile_collection_cashflows":
                identity = ("reconciliations", output["response"]["receipt_hash"])
                ledger = CollectionLedger(settings)
                rows = conn.execute("SELECT * FROM collection_evidence WHERE task_id=? AND case_id=? ORDER BY kind,identity", (task_id, request["case_id"])).fetchmany(10003)
                if len(rows) > 10002:
                    raise ValueError("extension_native_collection_budget")
                decoded = {(row["kind"], row["identity"]): ledger._decode(row) for row in rows}
                case = decoded[("case", request["case_id"])]
                flows = [body for (entry_kind, _), body in decoded.items() if entry_kind == "flow"]
                schedule = decoded[("schedule", request["schedule_id"])] if request["schedule_id"] else None
                expected = reconciliation_reference(case, flows, request, schedule)
                actual = decoded[("reconciliation", identity[1])]
                measured_values(actual, expected)
                if digest(actual) != identity[1] or {key: value for key, value in output["response"].items() if key != "receipt_hash"} != actual:
                    raise ValueError("extension_native_collection_result_mismatch")
                for body in [case, *flows, *request["coverage"], *([schedule] if schedule else []), *([request["maturity"]] if request["maturity"] else [])]:
                    for key, hash_key in (("source_artifact_id", "source_artifact_hash"), ("terms_artifact_id", "terms_artifact_hash"), ("policy_artifact_id", "policy_artifact_hash"), ("basis_artifact_id", "basis_artifact_hash")):
                        if key in body:
                            ledger._source(conn, task_id, body[key], body[hash_key])
                artifact = conn.execute("SELECT * FROM task_artifacts WHERE task_id=? AND kind='collection_cashflow_reconciliation' AND content_hash=?", (task_id, identity[1])).fetchone()
                if artifact is None:
                    raise ValueError("extension_native_collection_artifact_unavailable")
                ledger._source(conn, task_id, artifact["id"], identity[1])
                values["reconciliations"].append({"receipt_hash": identity[1], "status": expected["status"], "visible_event_count": expected["visible_event_count"], "amounts": expected["amounts"], "observed_subtotals": expected["observed_subtotals"], "reasons": expected["reasons"]})
            elif kind == "prepare_collection_batch":
                identity = ("batches", request["batch_id"])
                service = CollectionService(settings)
                row = conn.execute("SELECT * FROM collection_batches WHERE task_id=? AND batch_id=?", (task_id, request["batch_id"])).fetchone()
                if row is None:
                    raise ValueError("extension_native_batch_unavailable")
                native = service.read(task_id, request["batch_id"], row["actor_id"])
                if native["request"] != request:
                    raise ValueError("extension_native_batch_binding_mismatch")
                expected_inputs = {key: native[key] for key in ("batch_id", "request_hash", "preview_hash")}
                rejected = _rejected_workflows(conn, task_id, output, expected_inputs, {"collection.queue_batch", "collection.execute_reference", "collection.cancel_batch"}, outputs, decisions, http_events)
                values["rejections"].extend(rejected)
                if native["status"] == "proposed" or not native["effects"]:
                    if rejected:
                        continue
                    raise ValueError("extension_native_batch_not_executed")
                measured_values(native["preview"], batch_preview_reference(request))
                for effect in native["effects"]:
                    _producer(execution or {}, bound_outputs or {}, "collection." + effect["operation"], effect["artifact_id"])
                values["batches"].append({"batch_id": request["batch_id"], "status": native["status"], "revision": native["revision"], "action_count": len(native["actions"]), "customer_contacted": False, "actual_cost_minor": None})
            elif kind == "publish_operations_schedule":
                identity = ("schedules", request["schedule"]["schedule_id"], request["schedule"]["revision"])
                claims = [item["response"]["id"] for item in outputs if item["kind"] == "claim_extension_role" and item["actor"] == output["actor"]]
                if len(set(claims)) != 1:
                    raise ValueError("extension_native_schedule_publisher_unbound")
                values["schedules"].append(_operations(conn, request, claims[0]))
            else:
                continue
            seen.add(identity)
    if not seen and not values["rejections"]:
        raise ValueError("extension_native_measurements_unavailable")
    return {"verified": True, "scope": "local_native_carriers_and_supported_reference_recomputation", "acceptance_claim": "not_established", "rejection_scope": "native_http_or_exact_governed_step_fact; no_unobserved_external_effect_claim", "value_sha256": digest(values), "measurement_count": len(seen), "rejection_count": len(values["rejections"]), "identities": identities}, values
