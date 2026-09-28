"""Authenticate native event evidence inside the existing decision worker."""

from dataclasses import dataclass
import hmac

from marvis.reference_decision.contracts import DecisionError
from marvis.reference_decision.event_contracts import (
    EventEvidenceReference,
    EventFeatureBinding,
    EventFeatureRecipe,
)
from marvis.risk_context.event_contracts import EventError, EventFeatureContract
from marvis.risk_context.event_repository import EventRepository
from marvis.risk_context.event_service import EventService
from marvis.risk_context.source_contracts import SourceError


@dataclass(frozen=True)
class EventMaterial:
    values: dict
    evidence: dict


def authorize_binding(
    settings, binding: EventFeatureBinding, actor_id, *, grant_id=None
):
    if not actor_id:
        raise DecisionError("event_actor_required", 403)
    try:
        source = EventRepository(settings).authorize_read(
            binding.task_id,
            binding.recipe.source_id,
            grant_id=grant_id or binding.authoring_grant_id,
            actor_id=actor_id,
        )
    except (EventError, SourceError) as exc:
        raise DecisionError(exc.code, exc.status) from exc
    recipe = binding.recipe
    if source.contract_hash != recipe.source_contract_hash:
        raise DecisionError("event_source_contract_mismatch", 409)
    if (
        not set(recipe.event_types) <= set(source.event_types)
        or source.entity_namespaces.get(recipe.focus_kind) != recipe.focus_namespace
        or any(
            f.operation == "sum" and f.field not in source.numeric_fields
            for f in recipe.window_features
        )
    ):
        raise DecisionError("event_recipe_outside_source", 422)


def check_reference(settings, manifest, reference, actor_id):
    frozen = manifest["configuration"].get("event_binding")
    if frozen is None:
        if reference is not None:
            raise DecisionError("unexpected_event_evidence")
        return None
    if reference is None:
        raise DecisionError("event_evidence_required")
    binding = EventFeatureBinding.model_validate(frozen)
    reference = EventEvidenceReference.model_validate(reference)
    if (
        reference.task_id != binding.task_id
        or EventFeatureRecipe.from_contract(reference.contract) != binding.recipe
    ):
        raise DecisionError("event_recipe_binding_mismatch", 409)
    authorize_binding(settings, binding, actor_id, grant_id=reference.grant_id)
    return reference


def materialize(settings, manifest, reference, actor_id):
    reference = check_reference(settings, manifest, reference, actor_id)
    if reference is None:
        return None
    try:
        evidence = EventService(settings).evidence(
            reference.task_id,
            reference.request_id,
            actor_id,
            reference.grant_id,
        )
    except (EventError, SourceError) as exc:
        raise DecisionError(exc.code, exc.status) from exc
    if not hmac.compare_digest(
        evidence["content_hash"], reference.expected_content_hash
    ):
        raise DecisionError("event_evidence_content_hash_mismatch", 409)
    result = evidence["receipt"]["result"]
    if EventFeatureContract.model_validate(result["contract"]) != reference.contract:
        raise DecisionError("event_application_context_mismatch", 409)
    fields = EventFeatureRecipe.from_contract(reference.contract).field_names
    if set(fields) != set(result["features"]):
        raise DecisionError("event_feature_schema_mismatch", 409)
    # Unknown coverage/availability never becomes a zero or an approving default.
    if result["status"] != "measured" or any(
        result["features"][name]["status"] != "measured" for name in fields
    ):
        raise DecisionError("event_features_unknown", 409)
    values = {name: result["features"][name]["value"] for name in fields}
    if any(
        type(value) is not int or not -(2**63) <= value < 2**63
        for value in values.values()
    ):
        raise DecisionError("event_feature_value_out_of_range", 409)
    return EventMaterial(
        values=values,
        evidence={
            "task_id": reference.task_id,
            "request_id": reference.request_id,
            "artifact_id": evidence["artifact_id"],
            "content_hash": evidence["content_hash"],
            "source_id": reference.contract.source_id,
            "contract_hash": result["contract_hash"],
            "snapshot_hash": result["snapshot_hash"],
            "availability_mode": result["availability_mode"],
            "decision_at": reference.contract.decision_at,
            "knowledge_cutoff": reference.contract.knowledge_cutoff,
            "source_assurance": result["source_assurance"],
            "fraud_or_identity_proof": False,
        },
    )
