"""Frozen event recipes and per-application references to native evidence."""

from typing import Literal

from pydantic import Field, model_validator

from marvis.risk_context.event_contracts import (
    Contract,
    EntityKind,
    EventFeatureContract,
    Hash,
    Id,
    ReferenceText,
    RelationFeature,
    WindowFeature,
)


class EventFeatureRecipe(Contract):
    schema_version: Literal["risk-event.features.v1"] = "risk-event.features.v1"
    source_id: Id
    source_contract_hash: Hash
    availability_mode: Literal["platform_observed", "retrospective_declared"]
    window_seconds: int = Field(ge=1, le=36500 * 86400)
    interval: Literal["(t-window,t]"] = "(t-window,t]"
    include_current_event: bool
    event_types: list[Id] = Field(min_length=1, max_length=64)
    focus_kind: EntityKind
    focus_namespace: Id
    window_features: list[WindowFeature] = Field(default_factory=list, max_length=100)
    relation_features: list[RelationFeature] = Field(
        default_factory=list, max_length=100
    )
    policy_ref: ReferenceText

    @model_validator(mode="after")
    def validate_recipe(self):
        # Reuse the event contract's semantics; these constants validate only
        # the recipe's shape and never become an application or evidence record.
        EventFeatureContract.model_validate(
            {
                **self.model_dump(exclude={"focus_namespace"}),
                "focus": {"namespace": self.focus_namespace, "token": "0" * 64},
                "decision_at": "2020-01-01T00:00:00Z",
                "knowledge_cutoff": "2020-01-01T00:00:00Z",
                "current_event_id": None
                if self.include_current_event
                else "schema-validation",
            }
        )
        if any(name.startswith("__marvis_") for name in self.field_names):
            raise ValueError("event names cannot overlap reserved model outputs")
        return self

    @property
    def field_names(self):
        return [f.name for f in [*self.window_features, *self.relation_features]]

    @classmethod
    def from_contract(cls, contract):
        return cls.model_validate(
            {
                **contract.model_dump(
                    exclude={
                        "focus",
                        "decision_at",
                        "knowledge_cutoff",
                        "current_event_id",
                    }
                ),
                "focus_namespace": contract.focus.namespace,
            }
        )


class EventFeatureBinding(Contract):
    task_id: Id
    authoring_grant_id: Id
    recipe: EventFeatureRecipe


class EventEvidenceReference(Contract):
    task_id: Id
    request_id: Id
    grant_id: Id
    expected_content_hash: Hash
    # The caller explicitly selects the subject, times and exclusion semantics.
    # Every field must equal the already signed, governed event receipt.
    contract: EventFeatureContract
