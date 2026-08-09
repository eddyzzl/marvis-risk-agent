from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from marvis.db_schema import connect, init_db
from marvis.governance import (
    ApprovalBindingError,
    ApprovalExpired,
    ApprovalState,
    ApprovalStateError,
    AuthorizationBinding,
    EffectExecutionState,
    ExecutionContext,
    GovernanceRepository,
    PrincipalInactive,
    canonical_payload_hash,
)
from marvis.orchestrator.contracts import (
    plan_fingerprint,
    plan_step_confirmation_fingerprint,
)
from marvis.repositories.plans import PlanRepository
from marvis.state_machine import ConflictError
from tests.schema_fixture_support import install_v1_plan_step_runs_predecessor


class _Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 7, 18, 8, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta) -> None:
        self.now += timedelta(**delta)


def _binding(**overrides) -> AuthorizationBinding:
    values = {
        "task_id": "task-1",
        "plan_id": "plan-1",
        "plan_revision": 3,
        "step_id": "step-adopt",
        "tool_ref": "strategy.adopt_strategy@1.0.0",
        "manifest_hash": "manifest-sha256",
        "policy_hash": "policy-sha256",
        "input_hash": "input-sha256",
        "evidence_hash": "evidence-sha256",
        "effect_target": {
            "kind": "strategy",
            "id": "strategy-7",
            "expected_status": "draft",
            "result_status": "adopted",
        },
    }
    values.update(overrides)
    return AuthorizationBinding(**values)


def _grant(repo: GovernanceRepository, *, binding=None, ttl_seconds=900):
    principal = repo.create_local_principal(display_name="本地策略人员")
    grant = repo.authorize_effect(
        binding or _binding(),
        principal=principal,
        reason="确认采用该策略版本",
        ttl_seconds=ttl_seconds,
    )
    assert grant.approval is not None
    return principal, grant


def _seed_plan_gate(db_path, inputs: dict) -> None:
    policy_json = json.dumps(
        {
            "schema_version": "tool-policy.v1",
            "human_decision_gate": "required",
            "effect_authorization": "required",
            "effect_target": {
                "kind": "strategy",
                "id_input": "strategy_id",
                "expected_statuses": ["draft"],
                "result_status": "adopted",
            },
        },
        separators=(",", ":"),
    )
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO plans(
                id, task_id, goal, source, autonomy_level, status, replan_count,
                created_at, updated_at
            ) VALUES ('plan-1', 'task-1', 'goal', 'template', 0, 'awaiting_confirm', 3,
                      '2026-07-18T00:00:00+00:00', '2026-07-18T00:00:00+00:00')
            """
        )
        conn.execute(
            """
            INSERT INTO plan_steps(
                id, plan_id, idx, title, tool_plugin, tool_name, tool_version,
                inputs_json, depends_on_json, post_checks_json,
                needs_confirmation, policy_json, status, confirmed
            ) VALUES (
                'step-adopt', 'plan-1', 0, 'adopt', 'strategy', 'adopt_strategy', '1.0.0',
                ?, '[]', '[]', 1,
                ?,
                'awaiting_confirm', 0
            )
            """,
            (json.dumps(inputs, ensure_ascii=False), policy_json),
        )


def test_migration_creates_governance_tables_and_server_derived_principal(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = GovernanceRepository(db_path)

    principal = repo.create_local_principal(display_name="本地策略人员")

    assert principal.id
    assert principal.kind == "local_session"
    assert principal.display_name == "本地策略人员"
    assert repo.get_local_principal(principal.id) == principal
    with connect(db_path) as conn:
        tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    assert {
        "local_principals",
        "decision_records",
        "approval_records",
        "effect_executions",
    } <= tables


def test_local_session_cookie_is_opaque_hashed_and_resolved_server_side(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = GovernanceRepository(db_path)

    session = repo.create_local_session(display_name="浏览器本地用户")

    assert len(session.token) >= 32
    assert session.token not in db_path.read_bytes().decode("latin1")
    assert repo.resolve_local_session(session.token).id == session.principal.id
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT session_token_hash FROM local_principals WHERE id = ?",
            (session.principal.id,),
        ).fetchone()
    assert row["session_token_hash"] != session.token


def test_local_session_resolution_stays_read_only_when_another_writer_is_busy(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    clock = _Clock()
    repo = GovernanceRepository(db_path, clock=clock)
    session = repo.create_local_session(display_name="浏览器本地用户")
    clock.advance(minutes=1)

    resolved = []
    errors = []

    def resolve() -> None:
        try:
            resolved.append(repo.resolve_local_session(session.token))
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    thread = threading.Thread(target=resolve)
    with connect(db_path) as writer:
        writer.execute("BEGIN IMMEDIATE")
        writer.execute(
            "UPDATE local_principals SET display_name = display_name WHERE id = ?",
            (session.principal.id,),
        )
        thread.start()
        thread.join(timeout=0.75)
        completed_while_busy = not thread.is_alive()

    thread.join(timeout=6)
    assert completed_while_busy, "session resolution waited for an unrelated writer"
    assert not thread.is_alive()
    assert errors == []
    assert [principal.id for principal in resolved] == [session.principal.id]

    refreshed = repo.resolve_local_session(session.token)
    assert refreshed.last_seen_at == clock.now.isoformat()


def test_local_session_touch_defers_when_delete_journal_commit_is_busy(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    clock = _Clock()
    repo = GovernanceRepository(db_path, clock=clock)
    session = repo.create_local_session(display_name="浏览器本地用户")
    original_last_seen = session.principal.last_seen_at
    clock.advance(minutes=1)

    with sqlite3.connect(db_path) as mode_conn:
        assert mode_conn.execute("PRAGMA journal_mode=DELETE").fetchone()[0] == "delete"

    reader = sqlite3.connect(db_path, isolation_level="DEFERRED")
    try:
        reader.execute("BEGIN")
        reader.execute(
            "SELECT id FROM local_principals WHERE id = ?",
            (session.principal.id,),
        ).fetchone()

        resolved = repo.resolve_local_session(session.token)
    finally:
        reader.rollback()
        reader.close()

    assert resolved.id == session.principal.id
    assert repo.get_local_principal(session.principal.id).last_seen_at == original_last_seen


def test_expired_local_session_denial_survives_delete_journal_commit_contention(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    clock = _Clock()
    repo = GovernanceRepository(db_path, clock=clock)
    session = repo.create_local_session(ttl_seconds=1)
    clock.advance(seconds=2)

    with sqlite3.connect(db_path) as mode_conn:
        assert mode_conn.execute("PRAGMA journal_mode=DELETE").fetchone()[0] == "delete"

    reader = sqlite3.connect(db_path, isolation_level="DEFERRED")
    try:
        reader.execute("BEGIN")
        reader.execute(
            "SELECT id FROM local_principals WHERE id = ?",
            (session.principal.id,),
        ).fetchone()

        with pytest.raises(PrincipalInactive, match="expired"):
            repo.resolve_local_session(session.token)
    finally:
        reader.rollback()
        reader.close()

    assert repo.get_local_principal(session.principal.id).status == "active"


def test_expired_local_session_cannot_resolve_or_authorize(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    clock = _Clock()
    repo = GovernanceRepository(db_path, clock=clock)
    session = repo.create_local_session(ttl_seconds=1)
    clock.advance(seconds=2)

    with pytest.raises(PrincipalInactive, match="expired"):
        repo.resolve_local_session(session.token)

    assert repo.get_local_principal(session.principal.id).status == "expired"


def test_migration_backfills_pre_v5_confirmation_gates_as_required(tmp_path):
    db_path = tmp_path / "app.sqlite"
    with connect(db_path) as conn:
        install_v1_plan_step_runs_predecessor(conn)
        conn.execute(
            "CREATE TABLE plan_steps (id TEXT PRIMARY KEY, needs_confirmation INTEGER NOT NULL)"
        )
        conn.execute("INSERT INTO plan_steps VALUES ('legacy-gate', 1)")
        conn.execute("INSERT INTO plan_steps VALUES ('legacy-auto', 0)")
        conn.execute("PRAGMA user_version = 4")

    init_db(db_path)

    with connect(db_path) as conn:
        rows = {
            row["id"]: row["policy_json"]
            for row in conn.execute(
                "SELECT id, policy_json FROM plan_steps ORDER BY id"
            ).fetchall()
        }
    assert '"human_decision_gate":"required"' in rows["legacy-gate"]
    assert '"human_decision_gate":"none"' in rows["legacy-auto"]


def test_decision_is_immutable_and_approval_binds_complete_execution_context(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = GovernanceRepository(db_path)
    principal, grant = _grant(repo)

    assert grant.decision.principal_id == principal.id
    assert grant.decision.decision == "approve"
    assert grant.decision.binding == _binding()
    assert grant.approval is not None
    assert grant.approval.state is ApprovalState.ISSUED
    assert grant.context == ExecutionContext(
        plan_id="plan-1",
        plan_revision=3,
        step_id="step-adopt",
        decision_id=grant.decision.id,
        approval_id=grant.approval.id,
        runtime_generation=repo.runtime_generation,
        human_decision_required=True,
        effect_authorization_required=True,
    )

    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        with connect(db_path) as conn:
            conn.execute(
                "UPDATE decision_records SET reason = 'tampered' WHERE id = ?",
                (grant.decision.id,),
            )
    assert repo.get_decision(grant.decision.id) == grant.decision


@pytest.mark.parametrize(
    "changes",
    [
        {
            "approval_id": "approval-1",
            "human_decision_required": False,
            "effect_authorization_required": True,
        },
        {"approval_id": None, "effect_authorization_required": True},
        {"approval_id": "approval-1", "effect_authorization_required": False},
    ],
)
def test_execution_context_rejects_inconsistent_requirement_flags(changes):
    values = {
        "plan_id": "plan-1",
        "plan_revision": 3,
        "step_id": "step-1",
        "decision_id": "decision-1",
        "approval_id": None,
        "runtime_generation": "runtime-1",
        "human_decision_required": True,
        "effect_authorization_required": False,
    }
    values.update(changes)

    with pytest.raises(ValueError):
        ExecutionContext(**values)


def test_rejected_decision_never_issues_effect_approval(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = GovernanceRepository(db_path)
    principal = repo.create_local_principal()

    grant = repo.record_decision(
        _binding(),
        principal=principal,
        decision="reject",
        reason="证据不足",
        issue_effect_approval=True,
    )

    assert grant.decision.decision == "reject"
    assert grant.approval is None
    assert grant.context is None


def test_human_decision_only_binding_does_not_require_effect_target(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = GovernanceRepository(db_path)
    principal = repo.create_local_principal()
    binding = _binding(effect_target={}, step_id="select-cutoff")

    grant = repo.record_decision(
        binding,
        principal=principal,
        decision="approve",
        reason="选择该阈值方案",
    )

    assert grant.decision.binding.effect_target == {}
    assert grant.approval is None
    assert grant.context == ExecutionContext(
        plan_id="plan-1",
        plan_revision=3,
        step_id="select-cutoff",
        decision_id=grant.decision.id,
        approval_id=None,
        runtime_generation=repo.runtime_generation,
    )
    assert repo.execution_context_for_binding(binding) == grant.context
    assert repo.verify_decision(grant.context, binding) == grant.decision
    assert repo.list_decisions_by_step("plan-1", "select-cutoff") == [grant.decision]
    assert repo.list_approvals_by_step("plan-1", "select-cutoff") == []


@pytest.mark.parametrize(
    ("context_change", "binding_change"),
    [
        ({"plan_id": "other-plan"}, {}),
        ({"plan_revision": 4}, {}),
        ({"step_id": "other-step"}, {}),
        ({"decision_id": "other-decision"}, {}),
        ({"runtime_generation": "stale-runtime"}, {}),
        ({}, {"task_id": "other-task"}),
        ({}, {"tool_ref": "strategy.select_cutoff@2.0.0"}),
        ({}, {"manifest_hash": "changed"}),
        ({}, {"policy_hash": "changed"}),
        ({}, {"input_hash": "changed"}),
        ({}, {"evidence_hash": "changed"}),
    ],
)
def test_verify_decision_rejects_wrong_context_or_complete_binding(
    tmp_path,
    context_change,
    binding_change,
):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = GovernanceRepository(db_path)
    principal = repo.create_local_principal()
    binding = _binding(effect_target={}, step_id="select-cutoff")
    grant = repo.record_decision(
        binding,
        principal=principal,
        decision="approve",
        reason="选择阈值",
    )
    assert grant.context is not None

    with pytest.raises(ApprovalBindingError):
        repo.verify_decision(
            replace(grant.context, **context_change),
            replace(binding, **binding_change),
        )


def test_verify_decision_rejects_superseded_or_inactive_human_decision(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    clock = _Clock()
    repo = GovernanceRepository(db_path, clock=clock)
    principal = repo.create_local_principal()
    binding = _binding(effect_target={}, step_id="select-cutoff")
    first = repo.record_decision(
        binding,
        principal=principal,
        decision="approve",
        reason="第一次选择",
    )
    clock.advance(seconds=1)
    second = repo.record_decision(
        binding,
        principal=principal,
        decision="approve",
        reason="复核后选择",
    )

    with pytest.raises(ApprovalBindingError, match="stale"):
        repo.verify_decision(first.context, binding)
    assert repo.verify_decision(second.context, binding) == second.decision

    repo.revoke_local_session(principal.id)
    assert repo.execution_context_for_binding(binding) is None
    with pytest.raises(ApprovalBindingError, match="no live"):
        repo.verify_decision(second.context, binding)


def test_latest_reject_supersedes_prior_approve_for_exact_binding(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    clock = _Clock()
    repo = GovernanceRepository(db_path, clock=clock)
    principal = repo.create_local_principal()
    binding = _binding(effect_target={}, step_id="select-cutoff")
    approved = repo.record_decision(
        binding,
        principal=principal,
        decision="approve",
        reason="先批准",
    )
    clock.advance(seconds=1)
    rejected = repo.record_decision(
        binding,
        principal=principal,
        decision="reject",
        reason="复核后拒绝",
    )

    assert rejected.context is None
    assert repo.execution_context_for_binding(binding) is None
    assert repo.execution_context_for("plan-1", "select-cutoff", 3) is None
    with pytest.raises(ApprovalBindingError, match="no live"):
        repo.verify_decision(approved.context, binding)


def test_latest_reject_blocks_still_issued_effect_approval(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    clock = _Clock()
    repo = GovernanceRepository(db_path, clock=clock)
    principal = repo.create_local_principal()
    binding = _binding()
    approved = repo.authorize_effect(
        binding,
        principal=principal,
        reason="先批准效果",
    )
    clock.advance(seconds=1)
    repo.record_decision(
        binding,
        principal=principal,
        decision="reject",
        reason="复核后撤回",
    )

    approval = repo.get_approval(approved.approval.id)
    assert approval.state is ApprovalState.REVOKED
    assert approval.revoke_reason == "human_rejected"
    assert repo.execution_context_for_binding(binding) is None
    with pytest.raises(ApprovalStateError, match="revoked"):
        repo.reserve_effect(approved.context, binding)


def test_reject_after_effect_reservation_revokes_and_blocks_dispatch(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    clock = _Clock()
    repo = GovernanceRepository(db_path, clock=clock)
    principal, approved = _grant(repo)
    execution = repo.reserve_effect(approved.context, _binding())

    clock.advance(seconds=1)
    rejected = repo.record_decision(
        _binding(),
        principal=principal,
        decision="reject",
        reason="派发前复核拒绝",
    )

    assert rejected.context is None
    approval = repo.get_approval(approved.approval.id)
    assert approval.state is ApprovalState.REVOKED
    assert approval.revoke_reason == "human_rejected"
    assert repo.get_effect_execution(execution.id).state is EffectExecutionState.PREPARED
    with pytest.raises(ApprovalStateError, match="no longer reserved"):
        repo.mark_effect_dispatched(
            execution.id,
            reservation_id=execution.reservation_id,
        )


def test_executor_can_reload_latest_nonexpired_issued_context(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = GovernanceRepository(db_path, runtime_generation="runtime-after-restart")
    _, first = _grant(repo)
    _, second = _grant(repo)

    loaded = repo.execution_context_for("plan-1", "step-adopt", 3)

    assert loaded is not None
    assert loaded.decision_id == second.decision.id
    assert loaded.approval_id == second.approval.id
    assert loaded.human_decision_required is True
    assert loaded.effect_authorization_required is True
    assert loaded.runtime_generation == "runtime-after-restart"
    assert repo.list_approvals_by_step("plan-1", "step-adopt") == [
        first.approval,
        second.approval,
    ]


def test_authorize_step_atomically_records_and_confirms_plan_gate(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = GovernanceRepository(db_path)
    principal = repo.create_local_principal()
    inputs = {"strategy_id": "strategy-7", "reason": "人工采用"}
    binding = _binding(input_hash=canonical_payload_hash(inputs))
    _seed_plan_gate(db_path, inputs)

    grant = repo.authorize_step(
        binding,
        principal=principal,
        reason="确认采用策略",
        issue_effect_approval=True,
    )

    assert grant.approval is not None
    with connect(db_path) as conn:
        step = conn.execute(
            "SELECT confirmed FROM plan_steps WHERE id = 'step-adopt'"
        ).fetchone()
    assert step["confirmed"] == 1
    assert repo.list_decisions_by_step("plan-1", "step-adopt") == [grant.decision]

    with pytest.raises(ApprovalStateError):
        repo.authorize_step(
            binding,
            principal=principal,
            reason="重复确认",
            issue_effect_approval=True,
        )
    assert len(repo.list_decisions_by_step("plan-1", "step-adopt")) == 1


def test_authorize_step_guards_raw_plan_inputs_while_binding_resolved_values(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = GovernanceRepository(db_path)
    principal = repo.create_local_principal()
    raw_inputs = {"strategy_id": "$ref:select.output.strategy_id"}
    resolved_inputs = {"strategy_id": "strategy-7"}
    _seed_plan_gate(db_path, raw_inputs)
    binding = _binding(input_hash=canonical_payload_hash(resolved_inputs))

    grant = repo.authorize_step(
        binding,
        principal=principal,
        reason="确认解析后的目标",
        issue_effect_approval=True,
        expected_input_hash=canonical_payload_hash(raw_inputs),
    )

    assert grant.approval is not None
    assert grant.approval.binding.input_hash == canonical_payload_hash(resolved_inputs)


def test_authorize_step_accepts_matching_plan_and_step_snapshots(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = GovernanceRepository(db_path)
    principal = repo.create_local_principal()
    inputs = {"strategy_id": "strategy-7", "reason": "人工采用"}
    _seed_plan_gate(db_path, inputs)
    plan = PlanRepository(db_path).load_plan("plan-1")
    step = next(item for item in plan.steps if item.id == "step-adopt")

    grant = repo.authorize_step(
        _binding(input_hash=canonical_payload_hash(inputs)),
        principal=principal,
        reason="确认采用策略",
        issue_effect_approval=True,
        expected_plan_status=plan.status.value,
        expected_plan_fingerprint=plan_fingerprint(plan),
        expected_step_fingerprint=plan_step_confirmation_fingerprint(
            step,
            confirmed=False,
        ),
    )

    assert grant.approval is not None
    assert repo.list_decisions_by_step("plan-1", "step-adopt") == [grant.decision]
    with connect(db_path) as conn:
        confirmed = conn.execute(
            "SELECT confirmed FROM plan_steps WHERE id = 'step-adopt'"
        ).fetchone()["confirmed"]
    assert confirmed == 1


def test_authorize_step_rejects_stale_plan_status_without_partial_decision(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = GovernanceRepository(db_path)
    principal = repo.create_local_principal()
    inputs = {"strategy_id": "strategy-7"}
    _seed_plan_gate(db_path, inputs)
    with connect(db_path) as conn:
        conn.execute("UPDATE plans SET status = 'running' WHERE id = 'plan-1'")

    with pytest.raises(ApprovalStateError, match="plan status"):
        repo.authorize_step(
            _binding(input_hash=canonical_payload_hash(inputs)),
            principal=principal,
            reason="过期状态不得授权",
            issue_effect_approval=True,
            expected_plan_status="awaiting_confirm",
        )

    assert repo.list_decisions_by_step("plan-1", "step-adopt") == []
    with connect(db_path) as conn:
        confirmed = conn.execute(
            "SELECT confirmed FROM plan_steps WHERE id = 'step-adopt'"
        ).fetchone()["confirmed"]
    assert confirmed == 0


def test_authorize_step_rejects_stale_target_step_fingerprint_atomically(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = GovernanceRepository(db_path)
    principal = repo.create_local_principal()
    inputs = {"strategy_id": "strategy-7"}
    _seed_plan_gate(db_path, inputs)
    plan = PlanRepository(db_path).load_plan("plan-1")
    step = next(item for item in plan.steps if item.id == "step-adopt")
    expected_step_fingerprint = plan_step_confirmation_fingerprint(
        step,
        confirmed=False,
    )
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE plan_steps SET title = 'concurrently changed' WHERE id = 'step-adopt'"
        )

    with pytest.raises(ApprovalBindingError, match="step snapshot changed"):
        repo.authorize_step(
            _binding(input_hash=canonical_payload_hash(inputs)),
            principal=principal,
            reason="过期步骤快照不得授权",
            issue_effect_approval=True,
            expected_step_fingerprint=expected_step_fingerprint,
        )

    assert repo.list_decisions_by_step("plan-1", "step-adopt") == []
    with connect(db_path) as conn:
        confirmed = conn.execute(
            "SELECT confirmed FROM plan_steps WHERE id = 'step-adopt'"
        ).fetchone()["confirmed"]
    assert confirmed == 0


def test_authorize_step_rejects_other_step_change_via_full_plan_fingerprint(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = GovernanceRepository(db_path)
    principal = repo.create_local_principal()
    inputs = {"strategy_id": "strategy-7"}
    _seed_plan_gate(db_path, inputs)
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO plan_steps(
                id, plan_id, idx, title, tool_plugin, tool_name, tool_version,
                inputs_json, depends_on_json, post_checks_json,
                needs_confirmation, policy_json, status, confirmed
            ) VALUES (
                'step-other', 'plan-1', 1, 'other', 'strategy', 'noop', '1.0.0',
                '{}', '[]', '[]', 0,
                '{"schema_version":"tool-policy.v1","human_decision_gate":"none","effect_authorization":"none"}',
                'pending', 0
            )
            """
        )
    plan = PlanRepository(db_path).load_plan("plan-1")
    expected_plan_fingerprint = plan_fingerprint(plan)
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE plan_steps SET title = 'other changed' WHERE id = 'step-other'"
        )

    with pytest.raises(ApprovalBindingError, match="plan snapshot changed"):
        repo.authorize_step(
            _binding(input_hash=canonical_payload_hash(inputs)),
            principal=principal,
            reason="整份计划已变化不得授权",
            issue_effect_approval=True,
            expected_plan_fingerprint=expected_plan_fingerprint,
        )

    assert repo.list_decisions_by_step("plan-1", "step-adopt") == []
    with connect(db_path) as conn:
        confirmed = conn.execute(
            "SELECT confirmed FROM plan_steps WHERE id = 'step-adopt'"
        ).fetchone()["confirmed"]
    assert confirmed == 0


@pytest.mark.parametrize("with_input_updates", [False, True])
def test_raw_plan_confirmation_cannot_flip_governed_step(
    tmp_path,
    with_input_updates,
):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    _seed_plan_gate(db_path, {"strategy_id": "strategy-7"})
    plans = PlanRepository(db_path)

    with pytest.raises(ConflictError, match="governed human-decision"):
        if with_input_updates:
            plans.confirm_step_with_inputs(
                "step-adopt",
                input_updates={"reason": "attempted bypass"},
            )
        else:
            plans.confirm_step("step-adopt")

    with connect(db_path) as conn:
        step = conn.execute(
            "SELECT confirmed, inputs_json FROM plan_steps WHERE id = 'step-adopt'"
        ).fetchone()
        decision_count = conn.execute(
            "SELECT COUNT(*) AS count FROM decision_records"
        ).fetchone()["count"]
    assert step["confirmed"] == 0
    assert json.loads(step["inputs_json"]) == {"strategy_id": "strategy-7"}
    assert decision_count == 0


@pytest.mark.parametrize("with_input_updates", [False, True])
def test_raw_plan_confirmation_fails_closed_on_malformed_policy(
    tmp_path,
    with_input_updates,
):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    _seed_plan_gate(db_path, {"strategy_id": "strategy-7"})
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE plan_steps SET policy_json = '{not-json' WHERE id = 'step-adopt'"
        )
    plans = PlanRepository(db_path)

    with pytest.raises(ConflictError, match="invalid persisted governance policy"):
        if with_input_updates:
            plans.confirm_step_with_inputs(
                "step-adopt",
                input_updates={"reason": "attempted bypass"},
            )
        else:
            plans.confirm_step("step-adopt")

    assert plans.is_step_confirmed("step-adopt") is False


def test_raw_plan_confirmation_keeps_legacy_policy_none_compatible(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    _seed_plan_gate(db_path, {"strategy_id": "strategy-7"})
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE plan_steps SET policy_json = ? WHERE id = 'step-adopt'",
            (
                json.dumps(
                    {
                        "schema_version": "tool-policy.v1",
                        "human_decision_gate": "none",
                        "effect_authorization": "none",
                    },
                    separators=(",", ":"),
                ),
            ),
        )

    PlanRepository(db_path).confirm_step("step-adopt")

    assert PlanRepository(db_path).is_step_confirmed("step-adopt") is True


@pytest.mark.parametrize(
    ("context_change", "binding_change"),
    [
        ({"plan_id": "other-plan"}, {}),
        ({"plan_revision": 4}, {}),
        ({"step_id": "other-step"}, {}),
        ({"decision_id": "other-decision"}, {}),
        ({"runtime_generation": "stale-runtime"}, {}),
        ({}, {"task_id": "other-task"}),
        ({}, {"tool_ref": "strategy.adopt_strategy@2.0.0"}),
        ({}, {"manifest_hash": "changed"}),
        ({}, {"policy_hash": "changed"}),
        ({}, {"input_hash": "changed"}),
        ({}, {"evidence_hash": "changed"}),
        ({}, {"effect_target": {"kind": "strategy", "id": "other"}}),
    ],
)
def test_reservation_fails_closed_on_any_context_or_binding_mismatch(
    tmp_path,
    context_change,
    binding_change,
):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = GovernanceRepository(db_path)
    _, grant = _grant(repo)
    assert grant.context is not None

    with pytest.raises(ApprovalBindingError):
        repo.reserve_effect(
            replace(grant.context, **context_change),
            replace(_binding(), **binding_change),
        )

    assert repo.get_approval(grant.approval.id).state is ApprovalState.ISSUED
    assert repo.list_effect_executions(grant.approval.id) == []


def test_expired_approval_is_atomically_terminal_and_cannot_reserve(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    clock = _Clock()
    repo = GovernanceRepository(db_path, clock=clock)
    _, grant = _grant(repo, ttl_seconds=30)
    clock.advance(seconds=31)

    with pytest.raises(ApprovalExpired):
        repo.reserve_effect(grant.context, _binding())

    assert repo.get_approval(grant.approval.id).state is ApprovalState.EXPIRED


def test_begin_immediate_reservation_has_exactly_one_concurrent_winner(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = GovernanceRepository(db_path)
    _, grant = _grant(repo)
    barrier = threading.Barrier(2)
    results: list[object] = []
    lock = threading.Lock()

    def reserve() -> None:
        local_repo = GovernanceRepository(
            db_path,
            runtime_generation=grant.context.runtime_generation,
        )
        barrier.wait(timeout=3)
        try:
            result: object = local_repo.reserve_effect(grant.context, _binding())
        except Exception as exc:  # one state conflict is expected
            result = exc
        with lock:
            results.append(result)

    threads = [threading.Thread(target=reserve) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    winners = [item for item in results if not isinstance(item, Exception)]
    losers = [item for item in results if isinstance(item, Exception)]
    assert len(winners) == 1
    assert len(losers) == 1
    assert isinstance(losers[0], ApprovalStateError)
    assert repo.get_approval(grant.approval.id).state is ApprovalState.RESERVED
    assert len(repo.list_effect_executions(grant.approval.id)) == 1


def test_effect_ledger_requires_dispatch_before_commit_and_blocks_replay(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = GovernanceRepository(db_path)
    _, grant = _grant(repo)
    execution = repo.reserve_effect(grant.context, _binding())

    assert execution.state is EffectExecutionState.PREPARED
    with pytest.raises(ApprovalStateError):
        repo.mark_effect_committed(
            execution.id,
            reservation_id=execution.reservation_id,
            result_hash="result-sha256",
        )

    dispatched = repo.mark_effect_dispatched(
        execution.id,
        reservation_id=execution.reservation_id,
    )
    assert dispatched.state is EffectExecutionState.DISPATCHED
    committed = repo.mark_effect_committed(
        execution.id,
        reservation_id=execution.reservation_id,
        result_hash="result-sha256",
    )
    assert committed.state is EffectExecutionState.COMMITTED
    assert committed.result_hash == "result-sha256"
    assert repo.get_approval(grant.approval.id).state is ApprovalState.CONSUMED

    # A domain repository may have committed both ledgers atomically before the
    # Runner receives the successful result; Runner's acknowledgement is safe.
    assert repo.mark_effect_committed(
        execution.id,
        reservation_id=execution.reservation_id,
        result_hash="runner-output-hash-may-differ-from-domain-receipt",
    ) == committed

    with pytest.raises(ApprovalStateError):
        repo.reserve_effect(grant.context, _binding())


def test_startup_reconciliation_reissues_only_never_dispatched_preparations(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = GovernanceRepository(db_path)

    _, prepared_grant = _grant(repo, binding=_binding(step_id="prepared"))
    prepared = repo.reserve_effect(
        prepared_grant.context,
        _binding(step_id="prepared"),
    )

    _, dispatched_grant = _grant(repo, binding=_binding(step_id="dispatched"))
    dispatched = repo.reserve_effect(
        dispatched_grant.context,
        _binding(step_id="dispatched"),
    )
    repo.mark_effect_dispatched(
        dispatched.id,
        reservation_id=dispatched.reservation_id,
    )

    report = repo.reconcile_startup()

    assert report.released_prepared == (prepared.id,)
    assert report.marked_uncertain == (dispatched.id,)
    assert repo.get_approval(prepared_grant.approval.id).state is ApprovalState.ISSUED
    assert repo.get_effect_execution(prepared.id).released_at is not None
    assert repo.get_effect_execution(dispatched.id).state is EffectExecutionState.UNCERTAIN
    assert repo.get_approval(dispatched_grant.approval.id).state is ApprovalState.REVOKED

    # A later restart must not turn dispatched/uncertain work back into issued.
    second = repo.reconcile_startup()
    assert second.released_prepared == ()
    assert second.marked_uncertain == ()
    assert repo.get_approval(dispatched_grant.approval.id).state is ApprovalState.REVOKED
    with pytest.raises(ApprovalStateError):
        repo.reserve_effect(dispatched_grant.context, _binding(step_id="dispatched"))


def test_explicit_pre_dispatch_release_is_safe_but_post_dispatch_failure_is_uncertain(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = GovernanceRepository(db_path)

    _, safe_grant = _grant(repo, binding=_binding(step_id="safe"))
    safe_execution = repo.reserve_effect(safe_grant.context, _binding(step_id="safe"))
    released = repo.release_prepared_effect(
        safe_execution.id,
        reservation_id=safe_execution.reservation_id,
        reason="worker was not started",
    )
    assert released.released_at is not None
    assert repo.get_approval(safe_grant.approval.id).state is ApprovalState.ISSUED

    _, uncertain_grant = _grant(repo, binding=_binding(step_id="uncertain"))
    uncertain_execution = repo.reserve_effect(
        uncertain_grant.context,
        _binding(step_id="uncertain"),
    )
    repo.mark_effect_dispatched(
        uncertain_execution.id,
        reservation_id=uncertain_execution.reservation_id,
    )
    uncertain = repo.mark_effect_uncertain(
        uncertain_execution.id,
        reservation_id=uncertain_execution.reservation_id,
        reason="worker connection lost",
    )
    assert uncertain.state is EffectExecutionState.UNCERTAIN
    assert repo.get_approval(uncertain_grant.approval.id).state is ApprovalState.REVOKED
