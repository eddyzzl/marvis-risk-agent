"""Externally frozen, fixed native safety identities; no expected-value inputs."""
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

from .runtime_archive_connector import FrozenEventRequest
from .runtime_extensions_archive import FrozenExtensionBinding, Hash, Identity


class SafetyIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)


class FrozenGovernedAction(SafetyIdentity):
    action_index: int = Field(ge=1, le=500)
    plan_id: Identity
    step_id: Identity
    decision_id: Identity
    decision_record_sha256: Hash


class FrozenEventTime(SafetyIdentity):
    action_index: int = Field(ge=1, le=500)
    event: FrozenEventRequest


class FrozenDuplicateDecision(SafetyIdentity):
    kind: Literal["reference_decision"]
    action_indices: list[int] = Field(min_length=2, max_length=20)
    request_id: Identity
    slot: Literal["production", "shadow"]
    record_sha256: Hash


class FrozenDuplicateSource(SafetyIdentity):
    kind: Literal["historical_source_query"]
    action_indices: list[int] = Field(min_length=2, max_length=20)
    request_id: Identity
    record_sha256: Hash
    attempts_sha256: Hash


class FrozenReplayedApproval(SafetyIdentity):
    """One original approval and its later replays of the same native step."""
    approve_index: int = Field(ge=1, le=500)
    replay_indices: list[int] = Field(min_length=1, max_length=20)
    plan_id: Identity
    step_id: Identity


Duplicate = Annotated[FrozenDuplicateDecision | FrozenDuplicateSource, Field(discriminator="kind")]


class FrozenSafetyBinding(FrozenExtensionBinding):
    before_manifest_sha256: Hash | None = None
    governed_actions: list[FrozenGovernedAction] = Field(default_factory=list, max_length=100)
    event_times: list[FrozenEventTime] = Field(default_factory=list, max_length=100)
    duplicate_operations: list[Duplicate] = Field(default_factory=list, max_length=100)
    boundary_action_indices: list[int] = Field(default_factory=list, max_length=100)
    recovery_action_indices: list[int] = Field(default_factory=list, max_length=100)
    replayed_approvals: list[FrozenReplayedApproval] = Field(default_factory=list, max_length=20)

    @model_serializer(mode="wrap")
    def preserve_prior_binding_wire(self, handler):
        value = handler(self)
        if not self.recovery_action_indices:
            value.pop("recovery_action_indices", None)
        if not self.replayed_approvals:
            value.pop("replayed_approvals", None)
        return value

    @model_validator(mode="after")
    def original_indices(self):
        actions = self.case.actions
        indices = [item.action_index for item in self.governed_actions] + [item.action_index for item in self.event_times] + self.boundary_action_indices + self.recovery_action_indices
        for item in self.replayed_approvals:
            if item.replay_indices != sorted(set(item.replay_indices)) or item.replay_indices[0] <= item.approve_index:
                raise ValueError("replays must follow their original approval in increasing order")
            indices += [item.approve_index, *item.replay_indices]
        for item in self.duplicate_operations:
            if (any(type(i) is not int for i in item.action_indices) or item.action_indices != sorted(set(item.action_indices))):
                raise ValueError("duplicate actions require distinct increasing original indices")
            indices += item.action_indices
        if any(type(i) is not int or not 1 <= i <= len(actions) for i in indices):
            raise ValueError("safety action must reference an existing original action")
        for item in self.governed_actions:
            if getattr(actions[item.action_index - 1], "kind", None) not in {"approve_step", "reject_step", "recover_governed_decision"}:
                raise ValueError("governed safety requires an original approve/reject action")
            action = actions[item.action_index - 1]
            if action.kind == "recover_governed_decision" and (
                    action.fault_source_action is None
                    or getattr(actions[action.fault_source_action - 1].fault, "kind", None) != "stale_step_cas_once"):
                raise ValueError("new governed recovery decision requires an exact preceding stale CAS refusal")
        for index in self.recovery_action_indices:
            action = actions[index - 1]
            source = getattr(action, "fault_source_action", None)
            from .runtime_process_faults import KINDS
            allowed = {"discard_success_response_once"} | (KINDS | {"stale_step_cas_once"} if action.kind == "recover_governed_decision" else {"stale_report_revision_once"})
            if action.kind == "recover_validation_execution":
                allowed = {"tool_timeout_once"}
            if action.kind == "recover_monitoring_execution":
                allowed = {"stale_monitoring_revision_once", "tool_timeout_once"}
            if action.kind not in {"recover_governed_decision", "recover_validation_report", "recover_validation_execution", "recover_monitoring_execution"} or source is None or source >= index or getattr(getattr(actions[source - 1], "fault", None), "kind", None) not in allowed or actions[source - 1].tool != action.tool:
                raise ValueError("recovery must bind the exact earlier original discarded-response fault")
        for item in self.event_times:
            action = actions[item.action_index - 1]
            if getattr(getattr(action, "extension_request", None), "kind", None) != "prepare_risk_event_features":
                raise ValueError("time safety requires an original event feature request")
        for item in self.duplicate_operations:
            kinds = {"submit_reference_decision", "replay_reference_decision"} if item.kind == "reference_decision" else {"execute_risk_source_query"}
            if any(getattr(getattr(actions[i - 1], "extension_request", None), "kind", None) not in kinds for i in item.action_indices):
                raise ValueError("duplicate scope does not match original fixed native operations")
        return self
