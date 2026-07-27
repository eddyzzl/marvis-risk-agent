from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ApprovalState(str, Enum):
    ISSUED = "issued"
    RESERVED = "reserved"
    CONSUMED = "consumed"
    EXPIRED = "expired"
    REVOKED = "revoked"


class EffectExecutionState(str, Enum):
    PREPARED = "prepared"
    DISPATCHED = "dispatched"
    COMMITTED = "committed"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class AuthorizationBinding:
    """The complete immutable surface a human effect approval authorizes.

    Only hashes are persisted for inputs/evidence; raw samples, model payloads,
    and report content must never enter the governance ledger.
    """

    task_id: str
    plan_id: str
    plan_revision: int
    step_id: str
    tool_ref: str
    manifest_hash: str
    policy_hash: str
    input_hash: str
    evidence_hash: str
    effect_target: dict[str, Any]

    def __post_init__(self) -> None:
        for name in (
            "task_id",
            "plan_id",
            "step_id",
            "tool_ref",
            "manifest_hash",
            "policy_hash",
            "input_hash",
            "evidence_hash",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must not be empty")
        if int(self.plan_revision) < 0:
            raise ValueError("plan_revision must be >= 0")
        if not isinstance(self.effect_target, dict):
            raise ValueError("effect_target must be an object")
        # Copy through canonical JSON so the frozen record cannot be changed by
        # mutating the caller's original nested dict after authorization.
        canonical_target = json.loads(
            json.dumps(
                self.effect_target,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
        object.__setattr__(self, "effect_target", canonical_target)


@dataclass(frozen=True)
class ExecutionContext:
    """Out-of-band proof of the human decision authorizing one plan step.

    ``approval_id`` is an additional one-shot authorization for a governed
    effect; it is not a substitute for the immutable ``DecisionRecord``.  The
    two requirement flags make that distinction explicit at every executor /
    runner seam instead of asking callers to infer it from a nullable id.
    """

    plan_id: str
    plan_revision: int
    step_id: str
    decision_id: str
    approval_id: str | None
    runtime_generation: str
    human_decision_required: bool = True
    effect_authorization_required: bool = False

    def __post_init__(self) -> None:
        for name in ("plan_id", "step_id", "decision_id", "runtime_generation"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must not be empty")
        if int(self.plan_revision) < 0:
            raise ValueError("plan_revision must be >= 0")
        if not isinstance(self.human_decision_required, bool):
            raise ValueError("human_decision_required must be a boolean")
        if not isinstance(self.effect_authorization_required, bool):
            raise ValueError("effect_authorization_required must be a boolean")
        if self.effect_authorization_required and not self.human_decision_required:
            raise ValueError("effect authorization requires a human decision")
        has_approval = self.approval_id is not None
        if has_approval and not str(self.approval_id).strip():
            raise ValueError("approval_id must not be empty")
        if has_approval != self.effect_authorization_required:
            raise ValueError(
                "approval_id is required if and only if effect authorization is required"
            )


@dataclass(frozen=True)
class LocalPrincipal:
    id: str
    kind: str
    display_name: str
    status: str
    created_at: str
    last_seen_at: str
    expires_at: str
    revoked_at: str | None = None


@dataclass(frozen=True)
class LocalSession:
    """One-time delivery of the opaque cookie value plus its principal.

    Only ``sha256(token)`` is stored. Callers must put ``token`` in a secure,
    HTTP-only local session cookie and discard their in-memory copy when done.
    """

    principal: LocalPrincipal
    token: str


@dataclass(frozen=True)
class DecisionRecord:
    id: str
    binding: AuthorizationBinding
    principal_id: str
    decision: str
    reason: str
    created_at: str


@dataclass(frozen=True)
class ApprovalRecord:
    id: str
    decision_id: str
    binding: AuthorizationBinding
    principal_id: str
    reason: str
    nonce: str
    state: ApprovalState
    issued_at: str
    expires_at: str
    reserved_at: str | None = None
    reservation_id: str | None = None
    consumed_at: str | None = None
    revoked_at: str | None = None
    revoke_reason: str | None = None


@dataclass(frozen=True)
class EffectExecution:
    id: str
    approval_id: str
    reservation_id: str
    runtime_generation: str
    state: EffectExecutionState
    prepared_at: str
    dispatched_at: str | None = None
    committed_at: str | None = None
    uncertain_at: str | None = None
    released_at: str | None = None
    release_reason: str | None = None
    uncertain_reason: str | None = None
    result_hash: str | None = None
    detail: dict[str, Any] | None = None

    @property
    def effect_execution_id(self) -> str:
        """Explicit worker/domain idempotency key alias."""

        return self.id


@dataclass(frozen=True)
class AuthorizationGrant:
    decision: DecisionRecord
    approval: ApprovalRecord | None
    context: ExecutionContext | None


@dataclass(frozen=True)
class ReconciliationReport:
    released_prepared: tuple[str, ...] = ()
    marked_uncertain: tuple[str, ...] = ()
    consumed_committed: tuple[str, ...] = ()
    revoked_orphans: tuple[str, ...] = ()
