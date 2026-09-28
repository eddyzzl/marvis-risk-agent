"""Native event mappings and signed server authority for the existing batch tool."""

from datetime import UTC, datetime
import hmac
import json

from pydantic import ValidationError

from marvis.db_schema import connect
from marvis.decision_twin._canonical import canonical_json, content_hash, parse_datetime
from marvis.decision_twin.event_batch_contracts import HistoricalNativeEventReference
from marvis.reference_decision.contracts import DecisionError
from marvis.reference_decision.event_binding import check_reference, materialize
from marvis.reference_decision.event_contracts import (
    EventEvidenceReference,
    EventFeatureRecipe,
)
from marvis.risk_context.event_contracts import EventError
from marvis.risk_context.event_repository import EventRepository
from marvis.risk_context.source_contracts import SourceError


MAX_REFERENCE_BYTES = 64_000
MAX_BATCH_REFERENCE_BYTES = 8_000_000


def map_row(row, mapping, decision_at, as_of):
    raw = row[mapping.reference_col]
    if not isinstance(raw, str) or not raw.strip():
        raise DecisionError("historical_native_event_reference_required")
    if len(raw.encode()) > MAX_REFERENCE_BYTES:
        raise DecisionError("historical_event_reference_budget_exceeded", 413)
    try:
        native = HistoricalNativeEventReference.model_validate(json.loads(raw))
    except (ValueError, ValidationError) as exc:
        raise DecisionError("historical_native_event_reference_invalid") from exc
    contract = native.contract
    namespace, token = (
        row[mapping.subject_namespace_col],
        row[mapping.subject_token_col],
    )
    if (
        type(namespace) is not str
        or type(token) is not str
        or contract.focus_kind != "subject"
        or contract.focus.namespace != namespace
        or contract.focus.token != token
    ):
        raise DecisionError("historical_event_subject_mismatch")
    if (
        parse_datetime(contract.decision_at, "event.decision_at") != decision_at
        or parse_datetime(contract.knowledge_cutoff, "event.knowledge_cutoff") > as_of
    ):
        raise DecisionError("historical_event_time_mismatch")
    return EventEvidenceReference(**native.model_dump(), grant_id=mapping.grant_id)


def check_scopes(settings, scopes, actor_id):
    if not actor_id:
        raise DecisionError("historical_event_actor_required", 403)
    repo = EventRepository(settings)
    for scope in scopes:
        try:
            source = repo.authorize_read(
                scope["task_id"],
                scope["source_id"],
                grant_id=scope["grant_id"],
                actor_id=actor_id,
            )
        except (EventError, SourceError) as exc:
            raise DecisionError(exc.code, exc.status) from exc
        if source.contract_hash != scope["source_contract_hash"]:
            raise DecisionError("historical_event_source_changed", 409)


class EventBatchAuthority:
    def __init__(self, material):
        self.material = material
        with connect(material.settings.db_path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS historical_event_intents(
                proposal_hash TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
                body TEXT NOT NULL, signature TEXT NOT NULL, created_at TEXT NOT NULL)""")
            conn.execute("""CREATE TRIGGER IF NOT EXISTS historical_event_intents_no_update
                BEFORE UPDATE ON historical_event_intents BEGIN SELECT RAISE(ABORT,'event batch intent is immutable'); END""")
            conn.execute("""CREATE TRIGGER IF NOT EXISTS historical_event_intents_no_delete
                BEFORE DELETE ON historical_event_intents
                WHEN EXISTS(SELECT 1 FROM tasks WHERE id=OLD.task_id)
                BEGIN SELECT RAISE(ABORT,'event batch intent is retained'); END""")

    def freeze(self, proposal, contract_hash, operation, scopes):
        material = self.material
        check_scopes(material.settings, scopes, material.actor_id)
        authorization = {"actor_id": material.actor_id, "scopes": scopes}
        proposal["event_authorization"] = authorization
        proposal.pop("proposal_hash", None)
        proposal["proposal_hash"] = content_hash(proposal)
        body = {
            "schema_version": "decision_twin.event_intent.v1",
            "task_id": material.task_id,
            "operation": operation,
            "contract_hash": contract_hash,
            "proposal_hash": proposal["proposal_hash"],
            **authorization,
        }
        with connect(material.settings.db_path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO historical_event_intents VALUES(?,?,?,?,?)",
                (
                    proposal["proposal_hash"],
                    material.task_id,
                    canonical_json(body),
                    material._signature(body),
                    datetime.now(UTC).isoformat(),
                ),
            )
        self.resolve(proposal["proposal_hash"], contract_hash, operation)
        return proposal

    def resolve(self, proposal_hash, contract_hash, operation):
        material = self.material
        with connect(material.settings.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM historical_event_intents WHERE proposal_hash=? AND task_id=?",
                (proposal_hash, material.task_id),
            ).fetchone()
        if row is None:
            raise DecisionError("historical_event_reviewed_proposal_required", 403)
        body = json.loads(row["body"])
        if (
            not hmac.compare_digest(row["signature"], material._signature(body))
            or body["task_id"] != material.task_id
            or body["proposal_hash"] != proposal_hash
            or body["operation"] != operation
            or body["contract_hash"] != contract_hash
        ):
            raise DecisionError("historical_event_intent_binding_failed", 409)
        check_scopes(material.settings, body["scopes"], body["actor_id"])
        material.actor_id = body["actor_id"]
        return {"actor_id": body["actor_id"], "scopes": body["scopes"]}


def prepare_mappings(material, contract, records, manifests):
    mappings = {mapping.scenario_kind: mapping for mapping in contract.event_mappings}
    expected = {
        scenario.kind
        for scenario in contract.scenarios
        if manifests[scenario.kind]["configuration"].get("event_binding")
    }
    if set(mappings) != expected:
        raise DecisionError("historical_event_mapping_required_for_each_bound_scenario")
    scopes = {}
    budget = 0
    for kind, mapping in mappings.items():
        manifest = manifests[kind]
        recipe = EventFeatureRecipe.model_validate(
            manifest["configuration"]["event_binding"]["recipe"]
        )
        checked_scope = False
        for record in records:
            reference = EventEvidenceReference.model_validate(
                record["event_refs"][kind]
            )
            budget += len(canonical_json(reference.model_dump()).encode())
            if budget > MAX_BATCH_REFERENCE_BYTES:
                raise DecisionError("historical_event_reference_budget_exceeded", 413)
            # No snapshot replay in the HTTP process. This authenticates the
            # current grant and exact package recipe before human review.
            if not checked_scope:
                check_reference(
                    material.settings, manifest, reference, material.actor_id
                )
                checked_scope = True
            elif (
                reference.task_id
                != manifest["configuration"]["event_binding"]["task_id"]
                or EventFeatureRecipe.from_contract(reference.contract) != recipe
            ):
                raise DecisionError("event_recipe_binding_mismatch", 409)
            scope = {
                "task_id": reference.task_id,
                "source_id": reference.contract.source_id,
                "source_contract_hash": reference.contract.source_contract_hash,
                "grant_id": mapping.grant_id,
            }
            scopes[content_hash(scope)] = scope
    return [scopes[key] for key in sorted(scopes)]


def evaluate_row(material, manifest, directory, record, scenario_kind):
    from marvis.reference_decision.evaluation import evaluate

    reference = EventEvidenceReference.model_validate(
        record["event_refs"][scenario_kind]
    )
    try:
        events = materialize(material.settings, manifest, reference, material.actor_id)
    except DecisionError as exc:
        if exc.code != "event_features_unknown":
            raise
        # Only authenticated, well-bound unknown coverage has a business fallback.
        # Integrity, wrong applicant, wrong time and permission failures abort.
        return {
            "action": {
                "type": manifest["configuration"]["failure_action"],
                "value": manifest["configuration"]["failure_action"],
                "reason_code": "EVENT_EVIDENCE_UNKNOWN",
                "stop": True,
            },
            "matched_rule_id": None,
            "score": None,
            "score_product": manifest["configuration"]["score_product"],
            "status": "fallback",
            "error_code": "event_features_unknown",
            "event_evidence": {
                "task_id": reference.task_id,
                "request_id": reference.request_id,
                "content_hash": reference.expected_content_hash,
                "contract_hash": reference.contract.contract_hash,
                "decision_at": reference.contract.decision_at,
                "knowledge_cutoff": reference.contract.knowledge_cutoff,
                "availability_mode": reference.contract.availability_mode,
                "status": "unknown",
            },
        }
    result = evaluate(manifest, directory, record["features"], event_material=events)
    result["status"], result["error_code"] = "decided", None
    result["event_evidence"]["status"] = "measured"
    return result
