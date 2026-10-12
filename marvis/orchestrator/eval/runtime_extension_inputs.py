"""Declared extension business requests, without routes or authority material.

These are execution inputs, never expected answers. Credentials are configured
through the existing local administration path and cannot occur in this DTO.
Generated task/artifact/principal identities are resolved by the fixed journey
against its own prior HTTP results; declarations do not confer authority.
"""
from typing import Annotated, Literal
from datetime import datetime
import json

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_serializer, model_validator

from marvis.collection.batches import CollectionBatchRequest
from marvis.collection.contracts import CollectionCase, InstallmentSchedule, ReconciliationRequest
from marvis.collection.proposals import CollectionBusinessProposal
from marvis.collection.service import CashflowImport, CollectionMaterial
from marvis.operations.router import PublishScheduleRequest, RecheckRequest, TickRequest
from marvis.production_governance.router import (
    ActivatePromotionRequest, ClaimPrincipalRequest, CreatePromotionRequest,
    PromotionApprovalRequest, RollbackDeploymentRequest,
)
from marvis.reference_decision.contracts import DecisionRequest, PackageBuildRequest
from marvis.reference_decision.router import InstallRequest
from marvis.risk_context.event_contracts import EventSourceContract, EventSourceGrant
from marvis.risk_context.event_service import EventDatasetImport, EventRequest
from marvis.risk_context.source_contracts import AuthorizationBasis, SourceGrant, SourceProfile, SourceQuery


Identity = Annotated[str, Field(pattern=r"^[A-Za-z0-9_.:-]{1,160}$")]


class ExtensionInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)
    actor: Literal["maker", "checker", "admin"]

    @field_validator("request", mode="before", check_fields=False)
    @classmethod
    def public_json_request(cls, value):
        if isinstance(value, dict):
            # The public carrier is JSON; datetime strings follow the actual
            # HTTP DTO's JSON validation, without Python coercion of scalars.
            return TypeAdapter(cls.model_fields["request"].annotation).validate_json(json.dumps(value, allow_nan=False))
        return value

    @model_serializer(mode="wrap")
    def public_json(self, handler):
        def convert(value):
            if isinstance(value, datetime):
                return value.isoformat()
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            if isinstance(value, list):
                return [convert(item) for item in value]
            return value
        return convert(handler(self))

    @model_validator(mode="after")
    def declared_role(self):
        maker = {"build_decision_package", "request_production_promotion", "prepare_risk_source_query", "execute_risk_source_query", "prepare_risk_event_features", "declare_collection_material", "create_collection_case", "create_collection_schedule", "import_collection_cashflows", "reconcile_collection_cashflows", "propose_collection_batch", "prepare_collection_batch"}
        admin = {"install_decision_package", "activate_production_promotion", "rollback_production_deployment", "tick_operations", "register_historical_source_profile", "declare_source_authorization_basis", "grant_risk_source", "register_risk_event_source", "grant_risk_event_source"}
        kind = getattr(self, "kind", None)
        if ((kind in maker and self.actor != "maker") or (kind in admin and self.actor != "admin")
                or (kind == "review_production_promotion" and self.actor not in {"checker", "admin"})
                or (kind in {"publish_operations_schedule", "recheck_operations_period"} and self.actor not in {"maker", "admin"})):
            raise ValueError("extension actor must match the actual public API role boundary")
        return self


class ClaimExtensionRole(ExtensionInput):
    kind: Literal["claim_extension_role"]
    request: ClaimPrincipalRequest

    @model_validator(mode="after")
    def role(self):
        if self.request.role != self.actor:
            raise ValueError("declared actor role must match its local role claim")
        return self


class BuildDecisionPackage(ExtensionInput):
    kind: Literal["build_decision_package"]
    request: PackageBuildRequest


class RequestPromotion(ExtensionInput):
    kind: Literal["request_production_promotion"]
    request: CreatePromotionRequest


class ReviewPromotion(ExtensionInput):
    kind: Literal["review_production_promotion"]
    promotion_id: Identity
    request: PromotionApprovalRequest


class InstallDecisionPackage(ExtensionInput):
    kind: Literal["install_decision_package"]
    request: InstallRequest


class ActivatePromotion(ExtensionInput):
    kind: Literal["activate_production_promotion"]
    promotion_id: Identity
    request: ActivatePromotionRequest


class RollbackDeployment(ExtensionInput):
    kind: Literal["rollback_production_deployment"]
    environment: Identity
    deployment_id: Identity
    request: RollbackDeploymentRequest


class SubmitDecision(ExtensionInput):
    kind: Literal["submit_reference_decision", "replay_reference_decision"]
    slot: Literal["production", "shadow"] = "production"
    request: DecisionRequest


class PublishSchedule(ExtensionInput):
    kind: Literal["publish_operations_schedule"]
    request: PublishScheduleRequest


class TickOperations(ExtensionInput):
    kind: Literal["tick_operations"]
    request: TickRequest


class RecheckPeriod(ExtensionInput):
    kind: Literal["recheck_operations_period"]
    schedule_id: Identity
    request: RecheckRequest


class RegisterHistoricalProfile(ExtensionInput):
    kind: Literal["register_historical_source_profile"]
    request: SourceProfile

    @model_validator(mode="after")
    def historical_only(self):
        if self.request.mode != "historical_file" or self.request.endpoint is not None:
            raise ValueError("network credentials/endpoints require existing local administration")
        return self


class DeclareSourceBasis(ExtensionInput):
    kind: Literal["declare_source_authorization_basis"]
    request: AuthorizationBasis


class GrantRiskSource(ExtensionInput):
    kind: Literal["grant_risk_source"]
    request: SourceGrant


class PrepareSourceQuery(ExtensionInput):
    kind: Literal["prepare_risk_source_query"]
    request: SourceQuery


class ExecuteSourceQuery(ExtensionInput):
    kind: Literal["execute_risk_source_query", "read_risk_source_evidence"]
    request_id: Identity


class RegisterEventSource(ExtensionInput):
    kind: Literal["register_risk_event_source"]
    request: EventSourceContract


class GrantEventSource(ExtensionInput):
    kind: Literal["grant_risk_event_source"]
    request: EventSourceGrant


class ImportEvents(ExtensionInput):
    kind: Literal["import_risk_event_dataset"]
    request: EventDatasetImport


class PrepareEventQuery(ExtensionInput):
    kind: Literal["prepare_risk_event_features"]
    request: EventRequest


class ReadEventQuery(ExtensionInput):
    kind: Literal["read_risk_event_evidence"]
    request_id: Identity
    grant_id: Identity


class DeclareCollectionMaterial(ExtensionInput):
    kind: Literal["declare_collection_material"]
    request: CollectionMaterial


class CreateCollectionCase(ExtensionInput):
    kind: Literal["create_collection_case"]
    request: CollectionCase


class CreateCollectionSchedule(ExtensionInput):
    kind: Literal["create_collection_schedule"]
    request: InstallmentSchedule


class ImportCollectionCashflows(ExtensionInput):
    kind: Literal["import_collection_cashflows"]
    request: CashflowImport


class ReconcileCollection(ExtensionInput):
    kind: Literal["reconcile_collection_cashflows"]
    request: ReconciliationRequest


class ProposeCollectionBatch(ExtensionInput):
    kind: Literal["propose_collection_batch"]
    request: CollectionBusinessProposal


class PrepareCollectionBatch(ExtensionInput):
    kind: Literal["prepare_collection_batch"]
    request: CollectionBatchRequest


class ReadCollectionEffect(ExtensionInput):
    kind: Literal["read_collection_effects"]
    batch_id: Identity


ExtensionBusinessInput = Annotated[
    ClaimExtensionRole | BuildDecisionPackage | RequestPromotion | ReviewPromotion
    | InstallDecisionPackage | ActivatePromotion | RollbackDeployment | SubmitDecision
    | PublishSchedule | TickOperations | RecheckPeriod | RegisterHistoricalProfile
    | DeclareSourceBasis | GrantRiskSource | PrepareSourceQuery | ExecuteSourceQuery
    | RegisterEventSource | GrantEventSource | ImportEvents | PrepareEventQuery
    | ReadEventQuery | DeclareCollectionMaterial | CreateCollectionCase
    | CreateCollectionSchedule | ImportCollectionCashflows | ReconcileCollection
    | ProposeCollectionBatch | PrepareCollectionBatch | ReadCollectionEffect,
    Field(discriminator="kind"),
]
