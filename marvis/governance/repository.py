from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from marvis.db_schema import connect
from marvis.governance.contracts import (
    ApprovalRecord,
    ApprovalState,
    AuthorizationBinding,
    AuthorizationGrant,
    DecisionRecord,
    EffectExecution,
    EffectExecutionState,
    ExecutionContext,
    LocalPrincipal,
    LocalSession,
    ReconciliationReport,
)
from marvis.governance.errors import (
    ApprovalBindingError,
    ApprovalExpired,
    ApprovalNotFound,
    ApprovalStateError,
    EffectExecutionNotFound,
    PrincipalInactive,
    PrincipalNotFound,
)


Clock = Callable[[], datetime]
DEFAULT_SESSION_TTL_SECONDS = 12 * 60 * 60
DEFAULT_APPROVAL_TTL_SECONDS = 15 * 60


def canonical_payload_hash(payload: Any) -> str:
    """Hash structured evidence without persisting its potentially sensitive body."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class GovernanceRepository:
    """SQLite source of truth for decisions and one-shot effect authorization.

    ``runtime_generation`` is server-owned and changes on application restart.
    It is copied into every prepared effect execution so the worker/domain
    transaction can audit the exact runtime that dispatched the effect.
    """

    def __init__(
        self,
        db_path: Path,
        *,
        clock: Clock | None = None,
        runtime_generation: str | None = None,
    ) -> None:
        self.db_path = db_path
        self._clock = clock or (lambda: datetime.now(UTC))
        self.runtime_generation = str(runtime_generation or uuid.uuid4().hex)
        if not self.runtime_generation.strip():
            raise ValueError("runtime_generation must not be empty")

    # ------------------------------------------------------------------
    # Local, server-derived browser sessions.
    # ------------------------------------------------------------------

    def create_local_session(
        self,
        *,
        display_name: str = "本地用户",
        ttl_seconds: int = DEFAULT_SESSION_TTL_SECONDS,
    ) -> LocalSession:
        display = str(display_name).strip()
        if not display:
            raise ValueError("display_name must not be empty")
        ttl = _positive_ttl(ttl_seconds, name="ttl_seconds")
        token = secrets.token_urlsafe(32)
        token_hash = _token_hash(token)
        principal_id = uuid.uuid4().hex
        now = self._now()
        expires_at = _plus_seconds(now, ttl)
        with connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO local_principals(
                    id, kind, display_name, session_token_hash, status,
                    created_at, last_seen_at, expires_at, revoked_at
                )
                VALUES (?, 'local_session', ?, ?, 'active', ?, ?, ?, NULL)
                """,
                (
                    principal_id,
                    display,
                    token_hash,
                    now,
                    now,
                    expires_at,
                ),
            )
            row = conn.execute(
                "SELECT * FROM local_principals WHERE id = ?",
                (principal_id,),
            ).fetchone()
        return LocalSession(principal=_principal_from_row(row), token=token)

    def create_local_principal(
        self,
        *,
        display_name: str = "本地用户",
        ttl_seconds: int = DEFAULT_SESSION_TTL_SECONDS,
    ) -> LocalPrincipal:
        """Compatibility helper for in-process callers/tests.

        HTTP code should call :meth:`create_local_session` so it can deliver the
        opaque token cookie. The token generated here is intentionally discarded
        and is never persisted in plaintext.
        """

        return self.create_local_session(
            display_name=display_name,
            ttl_seconds=ttl_seconds,
        ).principal

    def resolve_local_session(self, token: str) -> LocalPrincipal:
        raw_token = str(token)
        if not raw_token:
            raise PrincipalNotFound("local session token is missing")
        now = self._now()
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM local_principals WHERE session_token_hash = ?",
                (_token_hash(raw_token),),
            ).fetchone()
            if row is None:
                raise PrincipalNotFound("local session was not found")
            state = str(row["status"])
            if state != "active":
                raise PrincipalInactive(f"local session is {state}")
            if str(row["expires_at"]) <= now:
                conn.execute(
                    "UPDATE local_principals SET status = 'expired' WHERE id = ? AND status = 'active'",
                    (row["id"],),
                )
                conn.commit()
                raise PrincipalInactive("local session is expired")
            conn.execute(
                "UPDATE local_principals SET last_seen_at = ? WHERE id = ?",
                (now, row["id"]),
            )
            updated = conn.execute(
                "SELECT * FROM local_principals WHERE id = ?",
                (row["id"],),
            ).fetchone()
        return _principal_from_row(updated)

    def get_local_principal(self, principal_id: str) -> LocalPrincipal:
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM local_principals WHERE id = ?",
                (principal_id,),
            ).fetchone()
        if row is None:
            raise PrincipalNotFound(principal_id)
        return _principal_from_row(row)

    def revoke_local_session(self, principal_id: str) -> LocalPrincipal:
        now = self._now()
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM local_principals WHERE id = ?",
                (principal_id,),
            ).fetchone()
            if row is None:
                raise PrincipalNotFound(principal_id)
            conn.execute(
                """
                UPDATE local_principals
                   SET status = 'revoked', revoked_at = ?
                 WHERE id = ? AND status <> 'revoked'
                """,
                (now, principal_id),
            )
            conn.execute(
                """
                UPDATE approval_records
                   SET status = 'revoked', revoked_at = ?,
                       revoke_reason = 'principal_revoked'
                 WHERE principal_id = ? AND status IN ('issued', 'reserved')
                """,
                (now, principal_id),
            )
            updated = conn.execute(
                "SELECT * FROM local_principals WHERE id = ?",
                (principal_id,),
            ).fetchone()
        return _principal_from_row(updated)

    # ------------------------------------------------------------------
    # Human decisions and effect approvals.
    # ------------------------------------------------------------------

    def record_decision(
        self,
        binding: AuthorizationBinding,
        *,
        principal: LocalPrincipal,
        decision: str,
        reason: str,
        issue_effect_approval: bool = False,
        ttl_seconds: int = DEFAULT_APPROVAL_TTL_SECONDS,
    ) -> AuthorizationGrant:
        normalized_decision = _decision(decision)
        normalized_reason = _reason(reason)
        if issue_effect_approval and normalized_decision == "approve":
            _require_effect_target(binding)
            ttl = _positive_ttl(ttl_seconds, name="ttl_seconds")
        else:
            ttl = DEFAULT_APPROVAL_TTL_SECONDS
        now = self._now()
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._require_active_principal(conn, principal, now)
            return self._record_decision_tx(
                conn,
                binding=binding,
                principal_id=principal.id,
                decision=normalized_decision,
                reason=normalized_reason,
                issue_effect_approval=(
                    bool(issue_effect_approval) and normalized_decision == "approve"
                ),
                ttl_seconds=ttl,
                now=now,
            )

    def authorize_effect(
        self,
        binding: AuthorizationBinding,
        *,
        principal: LocalPrincipal,
        reason: str,
        ttl_seconds: int = DEFAULT_APPROVAL_TTL_SECONDS,
    ) -> AuthorizationGrant:
        _require_effect_target(binding)
        return self.record_decision(
            binding,
            principal=principal,
            decision="approve",
            reason=reason,
            issue_effect_approval=True,
            ttl_seconds=ttl_seconds,
        )

    def authorize_step(
        self,
        binding: AuthorizationBinding,
        *,
        principal: LocalPrincipal,
        reason: str,
        issue_effect_approval: bool,
        ttl_seconds: int = DEFAULT_APPROVAL_TTL_SECONDS,
        input_updates: dict[str, Any] | None = None,
        expected_input_hash: str | None = None,
        expected_step_status: str = "awaiting_confirm",
    ) -> AuthorizationGrant:
        """Atomically record approval and CAS-confirm its plan step.

        This is the endpoint/service seam: no crash can leave a confirmed step
        without its DecisionRecord/ApprovalRecord, or vice versa. Reviewed input
        changes (for example adoption reason) are merged before validating the
        binding's input hash and share the same transaction.
        """

        normalized_reason = _reason(reason)
        expected = str(expected_step_status).strip()
        if not expected:
            raise ValueError("expected_step_status must not be empty")
        if input_updates is not None and not isinstance(input_updates, dict):
            raise ValueError("input_updates must be an object")
        if issue_effect_approval:
            _require_effect_target(binding)
            ttl = _positive_ttl(ttl_seconds, name="ttl_seconds")
        else:
            ttl = DEFAULT_APPROVAL_TTL_SECONDS
        now = self._now()
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._require_active_principal(conn, principal, now)
            row = conn.execute(
                """
                SELECT s.id, s.plan_id, s.status, s.confirmed, s.inputs_json,
                       s.tool_plugin, s.tool_name, s.tool_version,
                       p.task_id, p.replan_count
                  FROM plan_steps AS s
                  JOIN plans AS p ON p.id = s.plan_id
                 WHERE s.id = ?
                """,
                (binding.step_id,),
            ).fetchone()
            if row is None:
                raise ApprovalBindingError(f"plan step not found: {binding.step_id}")
            _assert_step_binding(row, binding)
            if int(row["confirmed"] or 0):
                raise ApprovalStateError("plan step is already confirmed")
            if str(row["status"]) != expected:
                raise ApprovalStateError(
                    f"plan step status is {row['status']}, expected {expected}"
                )
            current_inputs = json.loads(str(row["inputs_json"] or "{}"))
            if not isinstance(current_inputs, dict):
                raise ApprovalBindingError("plan step inputs are not an object")
            merged_inputs = {**current_inputs, **(input_updates or {})}
            current_input_hash = canonical_payload_hash(merged_inputs)
            frozen_plan_input_hash = expected_input_hash or binding.input_hash
            if not secrets.compare_digest(current_input_hash, frozen_plan_input_hash):
                raise ApprovalBindingError(
                    "plan step inputs changed while authorizing"
                )

            grant = self._record_decision_tx(
                conn,
                binding=binding,
                principal_id=principal.id,
                decision="approve",
                reason=normalized_reason,
                issue_effect_approval=bool(issue_effect_approval),
                ttl_seconds=ttl,
                now=now,
            )
            cursor = conn.execute(
                """
                UPDATE plan_steps
                   SET inputs_json = ?, confirmed = 1
                 WHERE id = ? AND plan_id = ? AND status = ? AND confirmed = 0
                """,
                (
                    json.dumps(merged_inputs, ensure_ascii=False),
                    binding.step_id,
                    binding.plan_id,
                    expected,
                ),
            )
            if cursor.rowcount != 1:
                raise ApprovalStateError("plan step changed while authorizing")
        return grant

    def _record_decision_tx(
        self,
        conn: sqlite3.Connection,
        *,
        binding: AuthorizationBinding,
        principal_id: str,
        decision: str,
        reason: str,
        issue_effect_approval: bool,
        ttl_seconds: int,
        now: str,
    ) -> AuthorizationGrant:
        decision_id = uuid.uuid4().hex
        target_json = _target_json(binding.effect_target)
        conn.execute(
            """
            INSERT INTO decision_records(
                id, task_id, plan_id, plan_revision, step_id, tool_ref,
                principal_id, decision, reason, manifest_hash, policy_hash,
                input_hash, evidence_hash, effect_target_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                decision_id,
                binding.task_id,
                binding.plan_id,
                binding.plan_revision,
                binding.step_id,
                binding.tool_ref,
                principal_id,
                decision,
                reason,
                binding.manifest_hash,
                binding.policy_hash,
                binding.input_hash,
                binding.evidence_hash,
                target_json,
                now,
            ),
        )
        decision_record = DecisionRecord(
            id=decision_id,
            binding=binding,
            principal_id=principal_id,
            decision=decision,
            reason=reason,
            created_at=now,
        )
        if decision == "reject":
            # A reject is an immutable superseding decision.  Revoke every
            # still-live approval for the exact frozen binding in this same
            # writer transaction so a preparation reserved just before the
            # reject can never cross the dispatch boundary afterwards.
            self._revoke_exact_binding_approvals_tx(
                conn,
                binding=binding,
                now=now,
                reason="human_rejected",
            )
        if not issue_effect_approval:
            context = None
            if decision == "approve":
                context = ExecutionContext(
                    plan_id=binding.plan_id,
                    plan_revision=binding.plan_revision,
                    step_id=binding.step_id,
                    decision_id=decision_id,
                    approval_id=None,
                    runtime_generation=self.runtime_generation,
                    human_decision_required=True,
                    effect_authorization_required=False,
                )
            return AuthorizationGrant(
                decision=decision_record,
                approval=None,
                context=context,
            )

        approval_id = uuid.uuid4().hex
        nonce = secrets.token_urlsafe(24)
        expires_at = _plus_seconds(now, ttl_seconds)
        conn.execute(
            """
            INSERT INTO approval_records(
                id, decision_id, task_id, plan_id, plan_revision, step_id,
                tool_ref, principal_id, reason, manifest_hash, policy_hash,
                input_hash, evidence_hash, effect_target_json, nonce, status,
                issued_at, expires_at, reserved_at, reservation_id, consumed_at,
                revoked_at, revoke_reason
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'issued',
                    ?, ?, NULL, NULL, NULL, NULL, NULL)
            """,
            (
                approval_id,
                decision_id,
                binding.task_id,
                binding.plan_id,
                binding.plan_revision,
                binding.step_id,
                binding.tool_ref,
                principal_id,
                reason,
                binding.manifest_hash,
                binding.policy_hash,
                binding.input_hash,
                binding.evidence_hash,
                target_json,
                nonce,
                now,
                expires_at,
            ),
        )
        approval = ApprovalRecord(
            id=approval_id,
            decision_id=decision_id,
            binding=binding,
            principal_id=principal_id,
            reason=reason,
            nonce=nonce,
            state=ApprovalState.ISSUED,
            issued_at=now,
            expires_at=expires_at,
        )
        context = ExecutionContext(
            plan_id=binding.plan_id,
            plan_revision=binding.plan_revision,
            step_id=binding.step_id,
            decision_id=decision_id,
            approval_id=approval_id,
            runtime_generation=self.runtime_generation,
            human_decision_required=True,
            effect_authorization_required=True,
        )
        return AuthorizationGrant(
            decision=decision_record,
            approval=approval,
            context=context,
        )

    # ------------------------------------------------------------------
    # Executor reload and read APIs.
    # ------------------------------------------------------------------

    def execution_context_for(
        self,
        plan_id: str,
        step_id: str,
        plan_revision: int,
    ) -> ExecutionContext | None:
        """Compatibility lookup for callers that have not frozen a binding.

        New execution paths must use :meth:`execution_context_for_binding` so
        input, evidence, manifest, policy, and effect-target hashes are part of
        the lookup rather than merely checked later.
        """

        now = self._now()
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT d.*
                  FROM decision_records AS d
                  JOIN local_principals AS p ON p.id = d.principal_id
                 WHERE d.plan_id = ? AND d.step_id = ? AND d.plan_revision = ?
                   AND p.status = 'active' AND p.expires_at > ?
                 ORDER BY d.created_at DESC, d.rowid DESC
                 LIMIT 1
                """,
                (plan_id, step_id, int(plan_revision), now),
            ).fetchone()
            if row is not None:
                return self._execution_context_for_binding_tx(
                    conn,
                    _binding_from_row(row),
                    now,
                )
        return None

    def execution_context_for_binding(
        self,
        binding: AuthorizationBinding,
    ) -> ExecutionContext | None:
        """Return the newest live approval proof for the exact frozen binding.

        A human-only gate needs an active principal's immutable approve
        ``DecisionRecord``.  A binding with an effect target additionally needs
        that same decision's issued, non-expired one-shot ``ApprovalRecord``.
        """

        if not isinstance(binding, AuthorizationBinding):
            raise TypeError("binding must be an AuthorizationBinding")
        now = self._now()
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            return self._execution_context_for_binding_tx(conn, binding, now)

    def verify_decision(
        self,
        context: ExecutionContext,
        binding: AuthorizationBinding,
    ) -> DecisionRecord:
        """Verify a current human decision proof against the complete binding.

        The expected context is re-derived from immutable records on every
        invocation.  Consequently a context from another runtime generation,
        an older superseded decision, an inactive principal, or any changed
        binding field fails closed.
        """

        if not isinstance(context, ExecutionContext):
            raise ApprovalBindingError("execution context is invalid")
        if not isinstance(binding, AuthorizationBinding):
            raise ApprovalBindingError("authorization binding is invalid")
        now = self._now()
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            expected = self._execution_context_for_binding_tx(conn, binding, now)
            if expected is None:
                raise ApprovalBindingError(
                    "no live approved human decision matches the execution binding"
                )
            if context != expected:
                raise ApprovalBindingError(
                    "human decision context is stale or does not match the execution binding"
                )
            row = conn.execute(
                "SELECT * FROM decision_records WHERE id = ?",
                (context.decision_id,),
            ).fetchone()
            if row is None:  # defensive: decisions are immutable and cannot be deleted
                raise ApprovalBindingError("human decision record is missing")
            return _decision_from_row(row)

    def _execution_context_for_binding_tx(
        self,
        conn: sqlite3.Connection,
        binding: AuthorizationBinding,
        now: str,
    ) -> ExecutionContext | None:
        effect_required = bool(binding.effect_target)
        if effect_required:
            conn.execute(
                """
                UPDATE approval_records
                   SET status = 'expired'
                 WHERE plan_id = ? AND step_id = ? AND plan_revision = ?
                   AND status = 'issued' AND expires_at <= ?
                """,
                (binding.plan_id, binding.step_id, binding.plan_revision, now),
            )
        decision_row = self._latest_exact_decision_row_tx(
            conn,
            binding=binding,
            now=now,
        )
        if decision_row is None:
            return None
        # DecisionRecords are immutable, so reversal is represented by a newer
        # exact-binding record.  Never fall back to an older approval after a
        # reject (or after a newer effect approval becomes unusable).
        if str(decision_row["decision"]) != "approve":
            return None
        approval_id: str | None = None
        if effect_required:
            approval_row = conn.execute(
                """
                SELECT * FROM approval_records
                 WHERE decision_id = ? AND status = 'issued' AND expires_at > ?
                 ORDER BY issued_at DESC, rowid DESC
                 LIMIT 1
                """,
                (decision_row["id"], now),
            ).fetchone()
            if approval_row is None:
                return None
            if (
                str(approval_row["principal_id"])
                != str(decision_row["principal_id"])
                or _binding_from_row(approval_row) != binding
            ):
                return None
            approval_id = str(approval_row["id"])
        return ExecutionContext(
            plan_id=binding.plan_id,
            plan_revision=binding.plan_revision,
            step_id=binding.step_id,
            decision_id=str(decision_row["id"]),
            approval_id=approval_id,
            runtime_generation=self.runtime_generation,
            human_decision_required=True,
            effect_authorization_required=effect_required,
        )

    def _latest_exact_decision_row_tx(
        self,
        conn: sqlite3.Connection,
        *,
        binding: AuthorizationBinding,
        now: str,
    ) -> sqlite3.Row | None:
        rows = conn.execute(
            """
            SELECT d.*
              FROM decision_records AS d
              JOIN local_principals AS p ON p.id = d.principal_id
             WHERE d.plan_id = ? AND d.step_id = ? AND d.plan_revision = ?
               AND p.status = 'active' AND p.expires_at > ?
             ORDER BY d.created_at DESC, d.rowid DESC
            """,
            (binding.plan_id, binding.step_id, binding.plan_revision, now),
        ).fetchall()
        for decision_row in rows:
            if _binding_from_row(decision_row) == binding:
                return decision_row
        return None

    def _revoke_exact_binding_approvals_tx(
        self,
        conn: sqlite3.Connection,
        *,
        binding: AuthorizationBinding,
        now: str,
        reason: str,
    ) -> None:
        rows = conn.execute(
            """
            SELECT * FROM approval_records
             WHERE plan_id = ? AND step_id = ? AND plan_revision = ?
               AND status IN ('issued', 'reserved')
             ORDER BY issued_at, rowid
            """,
            (binding.plan_id, binding.step_id, binding.plan_revision),
        ).fetchall()
        approval_ids = [
            str(row["id"])
            for row in rows
            if _binding_from_row(row) == binding
        ]
        for approval_id in approval_ids:
            conn.execute(
                """
                UPDATE approval_records
                   SET status = 'revoked', revoked_at = ?, revoke_reason = ?
                 WHERE id = ? AND status IN ('issued', 'reserved')
                """,
                (now, reason, approval_id),
            )

    def get_decision(self, decision_id: str) -> DecisionRecord:
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM decision_records WHERE id = ?",
                (decision_id,),
            ).fetchone()
        if row is None:
            raise KeyError(decision_id)
        return _decision_from_row(row)

    def list_decisions_by_step(self, plan_id: str, step_id: str) -> list[DecisionRecord]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT * FROM decision_records
                 WHERE plan_id = ? AND step_id = ?
                 ORDER BY created_at, id
                """,
                (plan_id, step_id),
            ).fetchall()
        return [_decision_from_row(row) for row in rows]

    def get_approval(self, approval_id: str) -> ApprovalRecord:
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM approval_records WHERE id = ?",
                (approval_id,),
            ).fetchone()
        if row is None:
            raise ApprovalNotFound(approval_id)
        return _approval_from_row(row)

    def list_approvals_by_step(self, plan_id: str, step_id: str) -> list[ApprovalRecord]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT * FROM approval_records
                 WHERE plan_id = ? AND step_id = ?
                 ORDER BY issued_at, id
                """,
                (plan_id, step_id),
            ).fetchall()
        return [_approval_from_row(row) for row in rows]

    # ------------------------------------------------------------------
    # Effect reservation and crash-safe execution ledger.
    # ------------------------------------------------------------------

    def reserve_effect(
        self,
        context: ExecutionContext,
        binding: AuthorizationBinding,
    ) -> EffectExecution:
        _require_effect_target(binding)
        if (
            not isinstance(context, ExecutionContext)
            or not context.human_decision_required
            or not context.effect_authorization_required
            or context.approval_id is None
        ):
            raise ApprovalBindingError(
                "effect execution requires human-decision and effect-approval proof"
            )
        now = self._now()
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM approval_records WHERE id = ?",
                (context.approval_id,),
            ).fetchone()
            if row is None:
                raise ApprovalNotFound(context.approval_id)
            state = ApprovalState(str(row["status"]))
            if state is ApprovalState.ISSUED and str(row["expires_at"]) <= now:
                conn.execute(
                    "UPDATE approval_records SET status = 'expired' WHERE id = ? AND status = 'issued'",
                    (context.approval_id,),
                )
                conn.commit()
                raise ApprovalExpired(context.approval_id)
            if state is not ApprovalState.ISSUED:
                raise ApprovalStateError(
                    f"approval {context.approval_id} is {state.value}, expected issued"
                )
            expected_context = self._execution_context_for_binding_tx(
                conn,
                binding,
                now,
            )
            if expected_context != context:
                raise ApprovalBindingError(
                    "effect authorization context is stale or does not match the execution binding"
                )
            self._assert_execution_binding(row, context, binding)
            self._require_active_principal_id(conn, str(row["principal_id"]), now)
            active = conn.execute(
                """
                SELECT id FROM effect_executions
                 WHERE approval_id = ? AND released_at IS NULL
                """,
                (context.approval_id,),
            ).fetchone()
            if active is not None:
                raise ApprovalStateError(
                    f"approval already has active effect execution {active['id']}"
                )
            reservation_id = uuid.uuid4().hex
            execution_id = uuid.uuid4().hex
            cursor = conn.execute(
                """
                UPDATE approval_records
                   SET status = 'reserved', reserved_at = ?, reservation_id = ?
                 WHERE id = ? AND status = 'issued' AND expires_at > ?
                """,
                (now, reservation_id, context.approval_id, now),
            )
            if cursor.rowcount != 1:
                raise ApprovalStateError("approval changed while reserving")
            conn.execute(
                """
                INSERT INTO effect_executions(
                    id, approval_id, reservation_id, runtime_generation, status,
                    prepared_at, detail_json
                )
                VALUES (?, ?, ?, ?, 'prepared', ?, '{}')
                """,
                (
                    execution_id,
                    context.approval_id,
                    reservation_id,
                    context.runtime_generation,
                    now,
                ),
            )
            execution_row = conn.execute(
                "SELECT * FROM effect_executions WHERE id = ?",
                (execution_id,),
            ).fetchone()
        return _effect_from_row(execution_row)

    def mark_effect_dispatched(
        self,
        execution_id: str,
        *,
        reservation_id: str,
    ) -> EffectExecution:
        return self._transition_effect(
            execution_id,
            reservation_id=reservation_id,
            expected=EffectExecutionState.PREPARED,
            target=EffectExecutionState.DISPATCHED,
        )

    def mark_effect_committed(
        self,
        execution_id: str,
        *,
        reservation_id: str,
        result_hash: str | None = None,
    ) -> EffectExecution:
        now = self._now()
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute(
                "SELECT * FROM effect_executions WHERE id = ?",
                (execution_id,),
            ).fetchone()
            if current is None:
                raise EffectExecutionNotFound(execution_id)
            if not secrets.compare_digest(
                str(current["reservation_id"]),
                str(reservation_id),
            ):
                raise ApprovalBindingError("effect reservation id mismatch")
            if str(current["status"]) == EffectExecutionState.COMMITTED.value:
                existing_hash = _optional_str(current["result_hash"])
                # A domain repository can atomically commit a domain receipt
                # before ToolRunner receives the successful worker response.
                # That receipt hash is authoritative and need not equal the
                # runner's later generic output hash; the same reservation is
                # the idempotency fence.
                approval = conn.execute(
                    "SELECT status, reservation_id FROM approval_records WHERE id = ?",
                    (current["approval_id"],),
                ).fetchone()
                if approval is None:
                    raise ApprovalStateError("committed effect approval is missing")
                if str(approval["reservation_id"] or "") != reservation_id:
                    raise ApprovalBindingError("committed approval reservation mismatch")
                if str(approval["status"]) == ApprovalState.RESERVED.value:
                    conn.execute(
                        """
                        UPDATE approval_records
                           SET status = 'consumed', consumed_at = ?
                         WHERE id = ? AND status = 'reserved' AND reservation_id = ?
                        """,
                        (now, current["approval_id"], reservation_id),
                    )
                elif str(approval["status"]) != ApprovalState.CONSUMED.value:
                    raise ApprovalStateError(
                        "committed effect approval is not consumed"
                    )
                if existing_hash is None and result_hash:
                    conn.execute(
                        "UPDATE effect_executions SET result_hash = ? WHERE id = ? AND result_hash IS NULL",
                        (result_hash, execution_id),
                    )
                updated = conn.execute(
                    "SELECT * FROM effect_executions WHERE id = ?",
                    (execution_id,),
                ).fetchone()
                return _effect_from_row(updated)
            row = self._effect_for_transition(
                conn,
                execution_id,
                reservation_id,
                expected=EffectExecutionState.DISPATCHED,
            )
            cursor = conn.execute(
                """
                UPDATE effect_executions
                   SET status = 'committed', committed_at = ?, result_hash = ?
                 WHERE id = ? AND reservation_id = ? AND status = 'dispatched'
                   AND released_at IS NULL
                """,
                (now, result_hash or None, execution_id, reservation_id),
            )
            if cursor.rowcount != 1:
                raise ApprovalStateError("effect execution changed while committing")
            approval_cursor = conn.execute(
                """
                UPDATE approval_records
                   SET status = 'consumed', consumed_at = ?
                 WHERE id = ? AND status = 'reserved' AND reservation_id = ?
                """,
                (now, row["approval_id"], reservation_id),
            )
            if approval_cursor.rowcount != 1:
                raise ApprovalStateError("approval changed while committing effect")
            updated = conn.execute(
                "SELECT * FROM effect_executions WHERE id = ?",
                (execution_id,),
            ).fetchone()
        return _effect_from_row(updated)

    def mark_effect_uncertain(
        self,
        execution_id: str,
        *,
        reservation_id: str,
        reason: str,
    ) -> EffectExecution:
        normalized_reason = _reason(reason)
        now = self._now()
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = self._effect_for_transition(
                conn,
                execution_id,
                reservation_id,
                expected=EffectExecutionState.DISPATCHED,
            )
            conn.execute(
                """
                UPDATE effect_executions
                   SET status = 'uncertain', uncertain_at = ?, uncertain_reason = ?
                 WHERE id = ? AND reservation_id = ? AND status = 'dispatched'
                   AND released_at IS NULL
                """,
                (now, normalized_reason, execution_id, reservation_id),
            )
            cursor = conn.execute(
                """
                UPDATE approval_records
                   SET status = 'revoked', revoked_at = ?, revoke_reason = ?
                 WHERE id = ? AND status = 'reserved' AND reservation_id = ?
                """,
                (now, "effect_uncertain", row["approval_id"], reservation_id),
            )
            if cursor.rowcount != 1:
                raise ApprovalStateError("approval changed while marking effect uncertain")
            updated = conn.execute(
                "SELECT * FROM effect_executions WHERE id = ?",
                (execution_id,),
            ).fetchone()
        return _effect_from_row(updated)

    def release_prepared_effect(
        self,
        execution_id: str,
        *,
        reservation_id: str,
        reason: str,
    ) -> EffectExecution:
        normalized_reason = _reason(reason)
        now = self._now()
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = self._effect_for_transition(
                conn,
                execution_id,
                reservation_id,
                expected=EffectExecutionState.PREPARED,
            )
            conn.execute(
                """
                UPDATE effect_executions
                   SET released_at = ?, release_reason = ?
                 WHERE id = ? AND reservation_id = ? AND status = 'prepared'
                   AND released_at IS NULL
                """,
                (now, normalized_reason, execution_id, reservation_id),
            )
            approval = conn.execute(
                "SELECT expires_at FROM approval_records WHERE id = ?",
                (row["approval_id"],),
            ).fetchone()
            next_state = (
                ApprovalState.ISSUED.value
                if approval is not None and str(approval["expires_at"]) > now
                else ApprovalState.EXPIRED.value
            )
            cursor = conn.execute(
                """
                UPDATE approval_records
                   SET status = ?, reserved_at = NULL, reservation_id = NULL
                 WHERE id = ? AND status = 'reserved' AND reservation_id = ?
                """,
                (next_state, row["approval_id"], reservation_id),
            )
            if cursor.rowcount != 1:
                raise ApprovalStateError("approval changed while releasing preparation")
            updated = conn.execute(
                "SELECT * FROM effect_executions WHERE id = ?",
                (execution_id,),
            ).fetchone()
        return _effect_from_row(updated)

    def get_effect_execution(self, execution_id: str) -> EffectExecution:
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM effect_executions WHERE id = ?",
                (execution_id,),
            ).fetchone()
        if row is None:
            raise EffectExecutionNotFound(execution_id)
        return _effect_from_row(row)

    def list_effect_executions(self, approval_id: str) -> list[EffectExecution]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT * FROM effect_executions
                 WHERE approval_id = ? ORDER BY prepared_at, id
                """,
                (approval_id,),
            ).fetchall()
        return [_effect_from_row(row) for row in rows]

    def reconcile_startup(self) -> ReconciliationReport:
        """Conservatively reconcile effects left behind by a crashed runtime.

        Only a ``prepared`` execution proves the worker was never dispatched and
        may release its approval back to ``issued``. A ``dispatched`` execution
        becomes ``uncertain`` and its approval is revoked. Existing uncertain
        work remains terminal; it is never reissued without a new human decision.
        """

        now = self._now()
        released: list[str] = []
        uncertain: list[str] = []
        consumed: list[str] = []
        orphans: list[str] = []
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """
                SELECT * FROM effect_executions
                 WHERE released_at IS NULL
                 ORDER BY prepared_at, id
                """
            ).fetchall()
            for row in rows:
                execution_id = str(row["id"])
                approval_id = str(row["approval_id"])
                reservation_id = str(row["reservation_id"])
                state = EffectExecutionState(str(row["status"]))
                approval = conn.execute(
                    "SELECT status, expires_at, reservation_id FROM approval_records WHERE id = ?",
                    (approval_id,),
                ).fetchone()
                approval_reserved = (
                    approval is not None
                    and str(approval["status"]) == ApprovalState.RESERVED.value
                    and str(approval["reservation_id"] or "") == reservation_id
                )
                if state is EffectExecutionState.PREPARED:
                    conn.execute(
                        """
                        UPDATE effect_executions
                           SET released_at = ?, release_reason = 'startup_pre_dispatch_release'
                         WHERE id = ? AND status = 'prepared' AND released_at IS NULL
                        """,
                        (now, execution_id),
                    )
                    if approval_reserved:
                        next_state = (
                            ApprovalState.ISSUED.value
                            if str(approval["expires_at"]) > now
                            else ApprovalState.EXPIRED.value
                        )
                        conn.execute(
                            """
                            UPDATE approval_records
                               SET status = ?, reserved_at = NULL, reservation_id = NULL
                             WHERE id = ? AND status = 'reserved' AND reservation_id = ?
                            """,
                            (next_state, approval_id, reservation_id),
                        )
                        released.append(execution_id)
                    else:
                        orphans.append(approval_id)
                elif state is EffectExecutionState.DISPATCHED:
                    conn.execute(
                        """
                        UPDATE effect_executions
                           SET status = 'uncertain', uncertain_at = ?,
                               uncertain_reason = 'runtime_restarted_after_dispatch'
                         WHERE id = ? AND status = 'dispatched'
                        """,
                        (now, execution_id),
                    )
                    if approval_reserved:
                        conn.execute(
                            """
                            UPDATE approval_records
                               SET status = 'revoked', revoked_at = ?,
                                   revoke_reason = 'runtime_restarted_after_dispatch'
                             WHERE id = ? AND status = 'reserved' AND reservation_id = ?
                            """,
                            (now, approval_id, reservation_id),
                        )
                    uncertain.append(execution_id)
                elif state is EffectExecutionState.UNCERTAIN:
                    if approval_reserved:
                        conn.execute(
                            """
                            UPDATE approval_records
                               SET status = 'revoked', revoked_at = ?,
                                   revoke_reason = 'effect_uncertain'
                             WHERE id = ? AND status = 'reserved' AND reservation_id = ?
                            """,
                            (now, approval_id, reservation_id),
                        )
                elif state is EffectExecutionState.COMMITTED:
                    if approval is not None and str(approval["status"]) != "consumed":
                        conn.execute(
                            """
                            UPDATE approval_records
                               SET status = 'consumed', consumed_at = ?
                             WHERE id = ? AND status = 'reserved' AND reservation_id = ?
                            """,
                            (now, approval_id, reservation_id),
                        )
                        consumed.append(execution_id)

            orphan_rows = conn.execute(
                """
                SELECT a.id
                  FROM approval_records AS a
                 WHERE a.status = 'reserved'
                   AND NOT EXISTS (
                       SELECT 1 FROM effect_executions AS e
                        WHERE e.approval_id = a.id AND e.released_at IS NULL
                   )
                """
            ).fetchall()
            for orphan in orphan_rows:
                approval_id = str(orphan["id"])
                conn.execute(
                    """
                    UPDATE approval_records
                       SET status = 'revoked', revoked_at = ?,
                           revoke_reason = 'missing_effect_execution_ledger'
                     WHERE id = ? AND status = 'reserved'
                    """,
                    (now, approval_id),
                )
                orphans.append(approval_id)
        return ReconciliationReport(
            released_prepared=tuple(released),
            marked_uncertain=tuple(uncertain),
            consumed_committed=tuple(consumed),
            revoked_orphans=tuple(dict.fromkeys(orphans)),
        )

    def _transition_effect(
        self,
        execution_id: str,
        *,
        reservation_id: str,
        expected: EffectExecutionState,
        target: EffectExecutionState,
    ) -> EffectExecution:
        now = self._now()
        timestamp_column = {
            EffectExecutionState.DISPATCHED: "dispatched_at",
        }.get(target)
        if timestamp_column is None:
            raise ValueError(f"unsupported effect transition target: {target.value}")
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = self._effect_for_transition(
                conn,
                execution_id,
                reservation_id,
                expected=expected,
            )
            approval = conn.execute(
                "SELECT * FROM approval_records WHERE id = ?",
                (row["approval_id"],),
            ).fetchone()
            if (
                approval is None
                or str(approval["status"]) != ApprovalState.RESERVED.value
                or str(approval["reservation_id"] or "") != reservation_id
            ):
                raise ApprovalStateError("approval is no longer reserved for this effect")
            approval_binding = _binding_from_row(approval)
            latest_decision = self._latest_exact_decision_row_tx(
                conn,
                binding=approval_binding,
                now=now,
            )
            if (
                latest_decision is None
                or str(latest_decision["decision"]) != "approve"
                or str(latest_decision["id"]) != str(approval["decision_id"])
            ):
                raise ApprovalStateError(
                    "effect approval was superseded before dispatch"
                )
            cursor = conn.execute(
                f"UPDATE effect_executions SET status = ?, {timestamp_column} = ? "
                "WHERE id = ? AND reservation_id = ? AND status = ? "
                "AND released_at IS NULL",
                (target.value, now, execution_id, reservation_id, expected.value),
            )
            if cursor.rowcount != 1:
                raise ApprovalStateError("effect execution changed during transition")
            updated = conn.execute(
                "SELECT * FROM effect_executions WHERE id = ?",
                (execution_id,),
            ).fetchone()
        return _effect_from_row(updated)

    def _effect_for_transition(
        self,
        conn: sqlite3.Connection,
        execution_id: str,
        reservation_id: str,
        *,
        expected: EffectExecutionState,
    ) -> sqlite3.Row:
        row = conn.execute(
            "SELECT * FROM effect_executions WHERE id = ?",
            (execution_id,),
        ).fetchone()
        if row is None:
            raise EffectExecutionNotFound(execution_id)
        if not secrets.compare_digest(
            str(row["reservation_id"]),
            str(reservation_id),
        ):
            raise ApprovalBindingError("effect reservation id mismatch")
        if row["released_at"] is not None:
            raise ApprovalStateError("effect preparation was already released")
        state = EffectExecutionState(str(row["status"]))
        if state is not expected:
            raise ApprovalStateError(
                f"effect execution is {state.value}, expected {expected.value}"
            )
        return row

    def _assert_execution_binding(
        self,
        row: sqlite3.Row,
        context: ExecutionContext,
        binding: AuthorizationBinding,
    ) -> None:
        observed = _binding_from_row(row)
        mismatches: list[str] = []
        if context.plan_id != observed.plan_id:
            mismatches.append("context.plan_id")
        if int(context.plan_revision) != observed.plan_revision:
            mismatches.append("context.plan_revision")
        if context.step_id != observed.step_id:
            mismatches.append("context.step_id")
        if context.decision_id != str(row["decision_id"]):
            mismatches.append("context.decision_id")
        if context.approval_id != str(row["id"]):
            mismatches.append("context.approval_id")
        if context.runtime_generation != self.runtime_generation:
            mismatches.append("context.runtime_generation")
        if not context.human_decision_required:
            mismatches.append("context.human_decision_required")
        if not context.effect_authorization_required:
            mismatches.append("context.effect_authorization_required")
        for field in (
            "task_id",
            "plan_id",
            "plan_revision",
            "step_id",
            "tool_ref",
            "manifest_hash",
            "policy_hash",
            "input_hash",
            "evidence_hash",
            "effect_target",
        ):
            if getattr(binding, field) != getattr(observed, field):
                mismatches.append(field)
        if mismatches:
            raise ApprovalBindingError(
                "approval binding mismatch: " + ", ".join(mismatches)
            )

    def _require_active_principal(
        self,
        conn: sqlite3.Connection,
        principal: LocalPrincipal,
        now: str,
    ) -> None:
        if not isinstance(principal, LocalPrincipal):
            raise TypeError("principal must be a server-resolved LocalPrincipal")
        self._require_active_principal_id(conn, principal.id, now)

    def _require_active_principal_id(
        self,
        conn: sqlite3.Connection,
        principal_id: str,
        now: str,
    ) -> None:
        row = conn.execute(
            "SELECT status, expires_at FROM local_principals WHERE id = ?",
            (principal_id,),
        ).fetchone()
        if row is None:
            raise PrincipalNotFound(principal_id)
        state = str(row["status"])
        if state != "active":
            raise PrincipalInactive(f"local principal is {state}")
        if str(row["expires_at"]) <= now:
            conn.execute(
                "UPDATE local_principals SET status = 'expired' WHERE id = ? AND status = 'active'",
                (principal_id,),
            )
            conn.execute(
                """
                UPDATE approval_records
                   SET status = 'revoked', revoked_at = ?,
                       revoke_reason = 'principal_expired'
                 WHERE principal_id = ? AND status IN ('issued', 'reserved')
                """,
                (now, principal_id),
            )
            conn.commit()
            raise PrincipalInactive("local principal is expired")

    def _now(self) -> str:
        value = self._clock()
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).isoformat()


def _principal_from_row(row: sqlite3.Row) -> LocalPrincipal:
    return LocalPrincipal(
        id=str(row["id"]),
        kind=str(row["kind"]),
        display_name=str(row["display_name"]),
        status=str(row["status"]),
        created_at=str(row["created_at"]),
        last_seen_at=str(row["last_seen_at"]),
        expires_at=str(row["expires_at"]),
        revoked_at=_optional_str(row["revoked_at"]),
    )


def _decision_from_row(row: sqlite3.Row) -> DecisionRecord:
    return DecisionRecord(
        id=str(row["id"]),
        binding=_binding_from_row(row),
        principal_id=str(row["principal_id"]),
        decision=str(row["decision"]),
        reason=str(row["reason"]),
        created_at=str(row["created_at"]),
    )


def _approval_from_row(row: sqlite3.Row) -> ApprovalRecord:
    return ApprovalRecord(
        id=str(row["id"]),
        decision_id=str(row["decision_id"]),
        binding=_binding_from_row(row),
        principal_id=str(row["principal_id"]),
        reason=str(row["reason"]),
        nonce=str(row["nonce"]),
        state=ApprovalState(str(row["status"])),
        issued_at=str(row["issued_at"]),
        expires_at=str(row["expires_at"]),
        reserved_at=_optional_str(row["reserved_at"]),
        reservation_id=_optional_str(row["reservation_id"]),
        consumed_at=_optional_str(row["consumed_at"]),
        revoked_at=_optional_str(row["revoked_at"]),
        revoke_reason=_optional_str(row["revoke_reason"]),
    )


def _effect_from_row(row: sqlite3.Row) -> EffectExecution:
    detail = json.loads(str(row["detail_json"] or "{}"))
    return EffectExecution(
        id=str(row["id"]),
        approval_id=str(row["approval_id"]),
        reservation_id=str(row["reservation_id"]),
        runtime_generation=str(row["runtime_generation"]),
        state=EffectExecutionState(str(row["status"])),
        prepared_at=str(row["prepared_at"]),
        dispatched_at=_optional_str(row["dispatched_at"]),
        committed_at=_optional_str(row["committed_at"]),
        uncertain_at=_optional_str(row["uncertain_at"]),
        released_at=_optional_str(row["released_at"]),
        release_reason=_optional_str(row["release_reason"]),
        uncertain_reason=_optional_str(row["uncertain_reason"]),
        result_hash=_optional_str(row["result_hash"]),
        detail=detail if isinstance(detail, dict) else {},
    )


def _binding_from_row(row: sqlite3.Row) -> AuthorizationBinding:
    raw_target = row["effect_target_json"]
    target = json.loads(str(raw_target)) if raw_target not in (None, "") else {}
    return AuthorizationBinding(
        task_id=str(row["task_id"]),
        plan_id=str(row["plan_id"]),
        plan_revision=int(row["plan_revision"]),
        step_id=str(row["step_id"]),
        tool_ref=str(row["tool_ref"]),
        manifest_hash=str(row["manifest_hash"]),
        policy_hash=str(row["policy_hash"]),
        input_hash=str(row["input_hash"]),
        evidence_hash=str(row["evidence_hash"]),
        effect_target=target,
    )


def _assert_step_binding(row: sqlite3.Row, binding: AuthorizationBinding) -> None:
    mismatches: list[str] = []
    if str(row["plan_id"]) != binding.plan_id:
        mismatches.append("plan_id")
    if str(row["task_id"]) != binding.task_id:
        mismatches.append("task_id")
    if int(row["replan_count"]) != binding.plan_revision:
        mismatches.append("plan_revision")
    label = f"{row['tool_plugin']}.{row['tool_name']}"
    version = str(row["tool_version"] or "")
    stored_ref = f"{label}@{version}" if version else label
    # Older/template plans commonly leave ToolRef.version empty and let the
    # registry resolve the installed manifest version at execution time. The
    # service must still freeze that resolved version in the approval. An
    # explicitly pinned plan version, however, must match exactly.
    tool_matches = (
        stored_ref == binding.tool_ref
        if version
        else binding.tool_ref == label or binding.tool_ref.startswith(f"{label}@")
    )
    if not tool_matches:
        mismatches.append("tool_ref")
    if mismatches:
        raise ApprovalBindingError(
            "approval binding mismatch: " + ", ".join(mismatches)
        )


def _target_json(target: dict[str, Any]) -> str | None:
    if not target:
        return None
    return json.dumps(
        target,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _require_effect_target(binding: AuthorizationBinding) -> None:
    if not binding.effect_target:
        raise ValueError("effect authorization requires a non-empty effect_target")


def _decision(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in {"approve", "reject"}:
        raise ValueError("decision must be approve or reject")
    return normalized


def _reason(value: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError("reason must not be empty")
    return normalized


def _positive_ttl(value: int, *, name: str) -> int:
    ttl = int(value)
    if ttl <= 0:
        raise ValueError(f"{name} must be > 0")
    return ttl


def _plus_seconds(timestamp: str, seconds: int) -> str:
    return (datetime.fromisoformat(timestamp) + timedelta(seconds=seconds)).isoformat()


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)
