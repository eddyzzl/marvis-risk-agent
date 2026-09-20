"""Real persistence, dispatcher, API and crash windows; no caller-supplied success."""

from concurrent.futures import ThreadPoolExecutor
import json
import sqlite3
from threading import Event
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from marvis.db_schema import connect
from marvis.governance.outcome_verifiers import OutcomeProof
from marvis.orchestrator.contracts import PlanStatus, StepStatus, ReviewVerdict
from marvis.orchestrator.evidence import payload_hash
from marvis.orchestrator.reviewer import FinalReview
from marvis.repositories.hook_deliveries import HookDeliveryRepository
from marvis.routers.plans import router
from marvis.state_machine import ConflictError
from tests.test_orch_completion import CountingReviewer, _hooks
from tests.test_orch_executor import (
    FakeRunner,
    _executor,
    _ok,
    _plan,
    _repo_with_agent_task,
    _step,
)


def scenario(
    tmp_path, *, event="step.completed", fenced=False, register=True, reviewer=None,
    remaining_step=None,
):
    steps = [_step("step-1")]
    if remaining_step is not None:
        steps.append(remaining_step)
    repo, _, _ = _repo_with_agent_task(tmp_path, _plan(*steps))
    hooks, runner = _hooks(repo), FakeRunner([_ok({"echo": "original"}) for _ in steps])
    executor = _executor(
        repo, runner, hooks=hooks, reviewer=reviewer or CountingReviewer()
    )
    with connect(repo.db_path) as conn:
        conn.execute(
            "CREATE TABLE local_receipts(event_id TEXT,target_ref TEXT,generation INTEGER,binding TEXT,payload_hash TEXT,outcome TEXT,PRIMARY KEY(event_id,target_ref,generation))"
        )
    calls = []

    def producer(_event, payload):
        delivery = payload["_hook_delivery"]
        calls.append(delivery["generation"])
        plain = {
            key: value for key, value in payload.items() if key != "_hook_delivery"
        }
        with connect(repo.db_path) as conn:
            # The receipt and the synthetic effect are the same atomic row.
            # A negative receipt represents a producer fence, not mere absence.
            conn.execute(
                "INSERT INTO local_receipts VALUES (?,?,?,?,?,?)",
                (
                    payload["event_id"],
                    delivery["target_ref"],
                    delivery["generation"],
                    delivery["binding"],
                    payload_hash(plain),
                    "not_applied_fenced"
                    if fenced and delivery["generation"] == 1
                    else "applied",
                ),
            )
        if delivery["generation"] == 1 and len(calls) == 1:
            raise RuntimeError("producer committed then transport lost receipt")

    hooks.register_listener(event, producer, required=True, binding="local-producer.v1")
    executor.run("plan-1")
    assert repo.load_plan("plan-1").status == PlanStatus.FAILED
    targets = executor.reconciler._targets(repo.load_plan("plan-1"))
    target = next(item for item in targets if item.kind == "hook")

    def verifier(target, conn):
        bound = target.binding
        row = conn.execute(
            "SELECT * FROM local_receipts WHERE event_id=? AND target_ref=? AND generation=?",
            (bound["event_id"], bound["target_ref"], bound["generation"]),
        ).fetchone()
        if (
            not row
            or row["binding"] != bound["target"].get("binding")
            or row["payload_hash"] != bound["payload_hash"]
        ):
            return OutcomeProof("unknown", target.id, reason="原始生产者凭据不匹配。")
        return OutcomeProof(
            row["outcome"],
            target.id,
            target.id,
            payload_hash(dict(row)),
            "原生产者已核对。",
        )

    if register:
        executor.reconciler.verifiers.register(
            "hook", target.producer, "local-reference.v1", verifier
        )
    return repo, executor, runner, hooks, calls, target, verifier


@pytest.mark.parametrize("event", ["step.completed", "workflow.completed"])
def test_applied_receipt_recovers_only_completion_and_preserves_history(
    tmp_path, event
):
    repo, executor, runner, _, calls, target, _ = scenario(tmp_path, event=event)
    before = repo.list_step_runs("step-1")
    output_ref = repo.load_plan("plan-1").steps[0].output_ref
    result = executor.reconcile_execution("plan-1", target.id)
    assert result["outcome"] == "applied"
    plan = repo.load_plan("plan-1")
    assert plan.status == PlanStatus.DONE
    assert plan.steps[0].output_ref == output_ref
    assert repo.list_step_runs("step-1") == before
    assert len(runner.calls) == 1 and calls == [1]
    assert repo.unreconciled_step_ids(["step-1"]) == []
    assert not repo.workflow_completion_pending("plan-1")
    if event == "workflow.completed":
        assert any(item.type == "hook_completion_failed" for item in plan.loop_events)
    assert (
        executor.reconcile_execution("plan-1", target.id)["resolution_id"]
        == result["resolution_id"]
    )
    with connect(repo.db_path) as conn:
        row = conn.execute("SELECT * FROM execution_reconciliations").fetchone()
        assert json.loads(row["previous_state_json"])["status"] == "unknown"
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute("UPDATE execution_reconciliations SET outcome='unknown'")


def test_fenced_negative_allows_exactly_one_new_hook_generation(tmp_path):
    repo, executor, runner, _, calls, target, _ = scenario(tmp_path, fenced=True)
    result = executor.reconcile_execution("plan-1", target.id)
    assert result["outcome"] == "not_applied_fenced"
    assert repo.load_plan("plan-1").status == PlanStatus.DONE
    assert calls == [1, 2] and len(runner.calls) == 1
    row = HookDeliveryRepository(repo.db_path).list_deliveries(
        target.binding["event_id"]
    )[0]
    assert row["generation"] == 2 and row["retry_authorization_id"] is None
    with pytest.raises(ConflictError):
        HookDeliveryRepository(repo.db_path).finish(
            target.binding["event_id"],
            target.producer,
            status="succeeded",
            result={"ok": True},
            generation=1,
        )
    executor.reconcile_execution("plan-1", target.id)
    assert calls == [1, 2]
    with connect(repo.db_path) as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM reconciliation_consumptions").fetchone()[
                0
            ]
            == 1
        )


@pytest.mark.parametrize(
    "execution_completed,expected",
    [(True, PlanStatus.DONE), (False, PlanStatus.FAILED), (None, PlanStatus.REVIEW)],
)
def test_reconciliation_keeps_execution_and_business_verdict_separate(
    tmp_path, execution_completed, expected
):
    class Reviewer(CountingReviewer):
        def final_review(self, _plan, _outputs, _goal):
            return FinalReview(
                False, "Business criterion failed", [], goal_doubt=True,
                execution_completed=execution_completed,
                business_acceptance={"status": "failed"},
            )

    repo, executor, runner, _, calls, target, _ = scenario(
        tmp_path, reviewer=Reviewer()
    )
    before = repo.list_step_runs("step-1")
    result = executor.reconcile_execution("plan-1", target.id)
    assert result["outcome"] == "applied"
    assert repo.load_plan("plan-1").status == expected
    assert repo.list_step_runs("step-1") == before
    assert len(runner.calls) == 1 and calls == [1]


def test_missing_adapter_stays_blocked_and_cannot_retry_or_rollback(tmp_path):
    repo, executor, runner, _, calls, target, _ = scenario(tmp_path, register=False)
    assert executor.reconcile_execution("plan-1", target.id)["outcome"] == "unknown"
    assert repo.load_plan("plan-1").status == PlanStatus.FAILED
    with pytest.raises(ConflictError):
        repo.retry_failed_step("plan-1", "step-1")
    assert len(runner.calls) == 1 and calls == [1]
    assert (
        executor.reconciler.describe(repo.load_plan("plan-1"))["targets"][0][
            "supported"
        ]
        is False
    )


@pytest.mark.parametrize("change", ["goal", "input", "output", "target", "generation"])
def test_binding_changed_while_verifier_runs_is_rejected(tmp_path, change):
    repo, executor, _, _, _, target, verifier = scenario(tmp_path, register=False)

    def delayed(target, connection):
        proof = verifier(target, connection)
        with connect(repo.db_path) as writer:
            if change == "goal":
                writer.execute("UPDATE plans SET goal='revised' WHERE id='plan-1'")
            elif change == "input":
                writer.execute(
                    "UPDATE plan_steps SET inputs_json=? WHERE id='step-1'",
                    (json.dumps({"changed": True}),),
                )
            elif change == "output":
                writer.execute(
                    "UPDATE plan_steps SET output_ref='metrics:step-1:v99' WHERE id='step-1'"
                )
            elif change == "target":
                targets = json.loads(
                    writer.execute(
                        "SELECT targets_json FROM hook_events WHERE event_id=?",
                        (target.binding["event_id"],),
                    ).fetchone()[0]
                )
                targets[0]["binding"] = "changed"
                writer.execute(
                    "UPDATE hook_events SET targets_json=? WHERE event_id=?",
                    (json.dumps(targets), target.binding["event_id"]),
                )
            else:
                writer.execute("UPDATE hook_deliveries SET generation=2")
        return proof

    executor.reconciler.verifiers.register(
        "hook", target.producer, "delayed.v1", delayed
    )
    with pytest.raises((ConflictError, ValueError, KeyError)):
        executor.reconcile_execution("plan-1", target.id)
    with connect(repo.db_path) as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM execution_reconciliations").fetchone()[0]
            == 0
        )


def test_verifiers_are_read_only_and_bad_binding_is_rejected(tmp_path):
    repo, executor, _, _, _, target, _ = scenario(tmp_path, register=False)

    def write_attempt(target, conn):
        conn.execute("DELETE FROM local_receipts")

    executor.reconciler.verifiers.register(
        "hook", target.producer, "bad.v1", write_attempt
    )
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        executor.reconcile_execution("plan-1", target.id)
    assert repo.load_plan("plan-1").status == PlanStatus.FAILED


def test_registered_verifier_cannot_return_receipt_for_different_target(tmp_path):
    repo, executor, _, _, _, target, _ = scenario(tmp_path, register=False)
    executor.reconciler.verifiers.register(
        "hook",
        target.producer,
        "bad-binding.v1",
        lambda target, conn: OutcomeProof(
            "applied", "another-target", "receipt", "hash"
        ),
    )
    with pytest.raises(ValueError, match="original execution binding"):
        executor.reconcile_execution("plan-1", target.id)
    assert repo.load_plan("plan-1").status == PlanStatus.FAILED


def test_concurrent_requests_cannot_execute_completion_twice(tmp_path):
    repo, executor, runner, _, calls, target, verifier = scenario(
        tmp_path, register=False
    )
    started, release = Event(), Event()

    def blocked(target, conn):
        started.set()
        assert release.wait(5)
        return verifier(target, conn)

    executor.reconciler.verifiers.register(
        "hook", target.producer, "delayed.v1", blocked
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(executor.reconcile_execution, "plan-1", target.id)
        assert started.wait(5)
        with pytest.raises(ConflictError):
            executor.reconcile_execution("plan-1", target.id)
        release.set()
        assert first.result()["outcome"] == "applied"
    assert repo.load_plan("plan-1").status == PlanStatus.DONE
    assert len(runner.calls) == 1 and calls == [1]


def test_repeated_request_resumes_crash_after_verification_before_completion(
    tmp_path, monkeypatch
):
    repo, executor, runner, _, calls, target, _ = scenario(tmp_path)
    original = executor.reconciler._finish_completion
    with monkeypatch.context() as patch:
        patch.setattr(
            executor.reconciler,
            "_finish_completion",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("power loss")),
        )
        with pytest.raises(RuntimeError):
            executor.reconcile_execution("plan-1", target.id)
    assert repo.load_plan("plan-1").status == PlanStatus.FAILED
    executor.reconciler._finish_completion = original
    executor.reconcile_execution("plan-1", target.id)
    assert repo.load_plan("plan-1").status == PlanStatus.DONE
    assert len(runner.calls) == 1 and calls == [1]


def test_restart_after_done_before_resolution_receipt_keeps_recoverable_get_action(
    tmp_path, monkeypatch
):
    repo, executor, runner, _, calls, target, _ = scenario(
        tmp_path, event="workflow.completed"
    )

    class PowerLoss(BaseException):
        pass

    with monkeypatch.context() as patch:
        patch.setattr(
            executor.reconciler,
            "_record_completion",
            lambda *args: (_ for _ in ()).throw(PowerLoss()),
        )
        with pytest.raises(PowerLoss):
            executor.reconcile_execution("plan-1", target.id)
    assert repo.load_plan("plan-1").status == PlanStatus.DONE
    assert repo.workflow_completion_pending("plan-1")
    restarted = _executor(
        repo, FakeRunner([]), hooks=_hooks(repo), reviewer=CountingReviewer()
    )
    client = _client_for(repo, restarted)
    payload = client.get("/api/plans/plan-1").json()["plan"]
    next_target = payload["reconciliation"]["targets"][0]
    assert next_target["supported"] is True
    result = client.post(
        "/api/plans/plan-1/reconcile", json={"target_id": next_target["id"]}
    )
    assert result.status_code == 200, result.text
    assert "failure_envelope" not in result.json()["plan"]
    assert repo.unreconciled_step_ids(["step-1"]) == []
    assert len(runner.calls) == 1 and calls == [1]


def test_task_purge_does_not_require_deleting_reconciliation_history(tmp_path):
    from marvis.repositories.tasks import TaskRepository

    repo, executor, _, _, _, target, _ = scenario(tmp_path)
    result = executor.reconcile_execution("plan-1", target.id)
    task_id = repo.load_plan("plan-1").task_id
    TaskRepository(repo.db_path).delete_task(task_id)
    with connect(repo.db_path) as conn:
        assert conn.execute(
            "SELECT 1 FROM execution_reconciliations WHERE id=?",
            (result["resolution_id"],),
        ).fetchone()
        assert not conn.execute("SELECT 1 FROM tasks WHERE id=?", (task_id,)).fetchone()


def test_api_only_accepts_server_derived_identity(tmp_path):
    repo, executor, runner, _, calls, target, _ = scenario(tmp_path)
    app = FastAPI()
    app.include_router(router)
    app.state.plan_repo, app.state.plan_executor = repo, executor
    app.state.settings = SimpleNamespace(db_path=repo.db_path)
    client = TestClient(app)
    before = client.get("/api/plans/plan-1").json()["plan"]
    assert before["reconciliation"]["targets"][0]["id"] == target.id
    for extra in (
        {"success": True},
        {"receipt": {}},
        {"verifier_id": "local-reference.v1"},
    ):
        assert (
            client.post(
                "/api/plans/plan-1/reconcile", json={"target_id": target.id, **extra}
            ).status_code
            == 422
        )
    assert (
        client.post(
            "/api/plans/plan-1/reconcile", json={"target_id": "forged"}
        ).status_code
        == 409
    )
    response = client.post("/api/plans/plan-1/reconcile", json={"target_id": target.id})
    assert response.status_code == 200, response.text
    assert response.json()["plan"]["status"] == "done"
    assert "failure_envelope" not in response.json()["plan"]
    assert calls == [1] and len(runner.calls) == 1


def _client_for(repo, executor):
    app = FastAPI()
    app.include_router(router)
    app.state.plan_repo, app.state.plan_executor = repo, executor
    app.state.settings = SimpleNamespace(db_path=repo.db_path)
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize("needs_confirmation", [False, True])
def test_reconciled_middle_step_exposes_explicit_continuation_without_replay(
    tmp_path, needs_confirmation
):
    repo, executor, runner, _, _, target, _ = scenario(
        tmp_path,
        remaining_step=_step("step-2", index=1, depends_on=["step-1"],
                             needs_confirmation=needs_confirmation),
    )
    client = _client_for(repo, executor)
    before = repo.list_step_runs("step-1")
    response = client.post("/api/plans/plan-1/reconcile", json={"target_id": target.id})
    assert response.status_code == 200, response.text
    plan = response.json()["plan"]
    continuation = plan["reconciliation"]["continuation"]
    assert continuation["remaining_step_ids"] == ["step-2"]
    assert len(runner.calls) == 1
    body = {"expected_plan_fingerprint": continuation["expected_plan_fingerprint"]}
    assert client.post("/api/plans/plan-1/run", json=body).status_code == 202
    restored = repo.load_plan("plan-1")
    assert restored.status == (
        PlanStatus.AWAITING_CONFIRM if needs_confirmation else PlanStatus.DONE
    )
    assert len(runner.calls) == (1 if needs_confirmation else 2)
    assert repo.list_step_runs("step-1") == before
    # A stale repeated action cannot replay either step or bypass the new gate.
    assert client.post("/api/plans/plan-1/run", json=body).status_code == 409
    assert len(runner.calls) == (1 if needs_confirmation else 2)


def test_continuation_rechecks_snapshot_after_acquiring_execution_lease(tmp_path):
    from marvis.orchestrator.contracts import plan_fingerprint
    from marvis.repositories.tasks import TaskRepository
    from marvis.routers.plans import _run_plan_job

    repo, executor, runner, _, _, target, _ = scenario(
        tmp_path, remaining_step=_step("step-2", index=1, depends_on=["step-1"]),
    )
    executor.reconcile_execution("plan-1", target.id)
    plan = repo.load_plan("plan-1")
    fingerprint = plan_fingerprint(plan)
    tasks = TaskRepository(repo.db_path)
    job_id = tasks.start_job(plan.task_id, "plan")
    assert executor.reconciler.describe(plan).get("continuation") is None
    with connect(repo.db_path) as conn:
        conn.execute("UPDATE plans SET goal='changed after enqueue' WHERE id=?", (plan.id,))
    with pytest.raises(ConflictError, match="快照"):
        _run_plan_job(job_id, repo.db_path, executor, plan.id, fingerprint)
    assert len(runner.calls) == 1
    assert not tasks.task_has_active_job(plan.task_id)


def test_http_task_lease_conflict_and_recovery_after_commit_crash(
    tmp_path, monkeypatch
):
    repo, executor, runner, _, calls, target, verifier = scenario(
        tmp_path, register=False
    )
    entered, release = Event(), Event()

    def paused(target, conn):
        entered.set()
        assert release.wait(10)
        return verifier(target, conn)

    executor.reconciler.verifiers.register("hook", target.producer, "paused.v1", paused)
    client = _client_for(repo, executor)
    original = executor.reconciler._finish_completion
    with monkeypatch.context() as patch:
        patch.setattr(
            executor.reconciler,
            "_finish_completion",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("crash after durable proof")
            ),
        )
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(
                client.post,
                "/api/plans/plan-1/reconcile",
                json={"target_id": target.id},
            )
            assert entered.wait(5)
            second = client.post(
                "/api/plans/plan-1/reconcile", json={"target_id": target.id}
            )
            assert second.status_code == 409
            release.set()
            assert first.result().status_code == 500
    executor.reconciler._finish_completion = original
    response = client.post("/api/plans/plan-1/reconcile", json={"target_id": target.id})
    assert response.status_code == 200, response.text
    assert response.json()["plan"]["status"] == "done"
    assert len(runner.calls) == 1 and calls == [1]
    with connect(repo.db_path) as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE status IN ('pending','running')"
            ).fetchone()[0]
            == 0
        )


@pytest.mark.parametrize("passed", [True, False])
def test_real_atomic_strategy_producer_restores_original_output_through_api_without_rerun(
    tmp_path, passed
):
    from tests.test_governed_producer_receipt import _case, _adopt
    from marvis.orchestrator.reconciliation import register_governed_outcome_verifier

    case = _case(tmp_path)
    original_output = _adopt(case)
    case.plans.finish_step_run(
        case.run_id,
        status="interrupted",
        error_kind="unknown_effect",
        error="host lost output",
    )
    plan = case.plans.load_plan("producer-plan")
    step = plan.steps[0]
    step.status, step.error = (
        StepStatus.FAILED,
        "output lost; explicit reconciliation required",
    )
    case.plans.update_step(step)
    case.plans.set_plan_status(plan.id, PlanStatus.FAILED)

    class Reviewer(CountingReviewer):
        def deterministic_check(self, step, output):
            return ReviewVerdict(
                "deterministic",
                passed,
                [] if passed else ["business check failed"],
                "now",
            )

    runner = FakeRunner([])
    executor = _executor(
        case.plans, runner, hooks=_hooks(case.plans), reviewer=Reviewer()
    )
    register_governed_outcome_verifier(executor.reconciler.verifiers, case.governance)
    client = _client_for(case.plans, executor)
    target = client.get("/api/plans/producer-plan").json()["plan"]["reconciliation"][
        "targets"
    ][0]
    assert target["supported"] is True
    response = client.post(
        "/api/plans/producer-plan/reconcile", json={"target_id": target["id"]}
    )
    assert response.status_code == 200, response.text
    assert response.json()["plan"]["status"] == ("done" if passed else "failed")
    assert response.json()["plan"]["steps"][0]["review_verdicts"][0]["passed"] is passed
    assert case.plans.load_step_output("producer-step") == original_output
    assert runner.calls == []
    run = case.plans.list_step_runs("producer-step")[0]
    assert run["id"] == case.run_id and run["status"] == "succeeded"
    assert run["invocation_contract"] == case.contract
    assert len(case.strategies.list_strategy_artifacts(case.strategy.id)) == 2


def test_real_strategy_fence_unblocks_explicit_retry_but_does_not_reuse_approval(
    tmp_path,
):
    from tests.test_governed_producer_receipt import _case, _reserve
    from marvis.governance.repository import GovernanceRepository
    from marvis.orchestrator.reconciliation import register_governed_outcome_verifier

    case = _case(tmp_path)
    _reserve(case)
    governance = GovernanceRepository(
        case.settings.db_path, runtime_generation="restart"
    )
    governance.reconcile_startup()
    case.plans.finish_step_run(
        case.run_id,
        status="interrupted",
        error_kind="unknown_effect",
        error="lost worker",
    )
    plan = case.plans.load_plan("producer-plan")
    step = plan.steps[0]
    step.status, step.error = (
        StepStatus.FAILED,
        "lost; explicit reconciliation required",
    )
    case.plans.update_step(step)
    case.plans.set_plan_status(plan.id, PlanStatus.FAILED)
    runner = FakeRunner([])
    executor = _executor(
        case.plans, runner, hooks=_hooks(case.plans), reviewer=CountingReviewer()
    )
    register_governed_outcome_verifier(executor.reconciler.verifiers, governance)
    from marvis.agent.plan_driver import PlanDriver

    driver = PlanDriver(case.plans, executor)
    target = executor.reconciler.describe(case.plans.load_plan(plan.id))["targets"][0]
    assert (
        driver.reconcile_execution(plan.id, target["id"])["outcome"]
        == "not_applied_fenced"
    )
    assert case.plans.unreconciled_step_ids([step.id]) == []
    assert (
        case.plans.latest_failed_step_run_error_kind(step.id) == "execution_not_applied"
    )
    assert case.plans.retry_failed_step(plan.id, step.id) == [step.id]
    assert not case.plans.is_step_confirmed(step.id)
    assert governance.get_approval(case.grant.approval.id).state.value == "revoked"
    assert runner.calls == []


def test_upgrade_37_preserves_legacy_effect_without_inventing_invocation(
    tmp_path, monkeypatch
):
    import marvis.db_schema as schema
    from tests.test_governed_producer_receipt import _case

    with monkeypatch.context() as patch:
        patch.setattr(
            schema,
            "_MIGRATIONS",
            [item for item in schema._MIGRATIONS if item[0] <= 37],
        )
        case = _case(tmp_path)
        with connect(case.settings.db_path) as conn:
            conn.execute(
                "INSERT INTO effect_executions(id,approval_id,reservation_id,runtime_generation,status,prepared_at,detail_json) VALUES (?,?,?,?,?,?,?)",
                (
                    "legacy-effect",
                    case.grant.approval.id,
                    "legacy-reservation",
                    "old-runtime",
                    "prepared",
                    "old-time",
                    '{"legacy_evidence_hash":"must-preserve"}',
                ),
            )
    schema.init_db(case.settings.db_path)
    with connect(case.settings.db_path) as conn:
        row = conn.execute(
            "SELECT * FROM effect_executions WHERE id='legacy-effect'"
        ).fetchone()
        assert row["invocation_id"] is None and row["invocation_contract_hash"] is None
        assert row["detail_json"] == '{"legacy_evidence_hash":"must-preserve"}'
    assert case.governance.verify_producer_outcome(case.run_id)["outcome"] == "unknown"


def test_workflow_checkpoint_crash_startup_and_http_resume_original_frozen_completion(
    tmp_path, monkeypatch
):
    from marvis.recovery import reclaim_running_plans
    from marvis.orchestrator.harness_state import HarnessState

    repo, tasks, _ = _repo_with_agent_task(tmp_path, _plan(_step("step-1")))
    hooks, runner, reviewer = (
        _hooks(repo),
        FakeRunner([_ok({"echo": "value"})]),
        CountingReviewer(),
    )
    effects = []
    hooks.register_listener(
        "workflow.completed",
        lambda event, payload: effects.append(payload),
        required=True,
    )
    executor = _executor(repo, runner, hooks=hooks, reviewer=reviewer)
    original = HookDeliveryRepository.store_checkpoint

    class PowerLoss(BaseException):
        pass

    def crash(self, identity, *args):
        if identity.startswith("workflow:"):
            raise PowerLoss()
        return original(self, identity, *args)

    with monkeypatch.context() as patch:
        patch.setattr(HookDeliveryRepository, "store_checkpoint", crash)
        with pytest.raises(PowerLoss):
            executor.run("plan-1")
    summary = repo.latest_plan_summary_ref("plan-1")
    assert reclaim_running_plans(repo, reviewer, hooks, HarnessState(repo), tasks) == 1
    assert repo.load_plan("plan-1").status == PlanStatus.FAILED
    client = _client_for(repo, executor)
    payload = client.get("/api/plans/plan-1").json()["plan"]
    assert payload["failure_envelope"]["retryable"] is False
    target = payload["reconciliation"]["targets"][0]
    monkeypatch.setattr(
        reviewer,
        "final_review",
        lambda *args: (_ for _ in ()).throw(AssertionError("must reuse frozen review")),
    )
    response = client.post(
        "/api/plans/plan-1/reconcile", json={"target_id": target["id"]}
    )
    assert response.status_code == 200, response.text
    assert response.json()["plan"]["status"] == "done"
    assert repo.latest_plan_summary_ref("plan-1") == summary
    assert len(effects) == 1 and len(runner.calls) == 1


@pytest.mark.parametrize("applied", [True, False])
def test_cancelled_unknown_tool_can_be_verified_without_resuming_until_explicit_action(
    tmp_path, applied
):
    from tests.test_governed_producer_receipt import _case, _adopt, _reserve
    from marvis.governance.repository import GovernanceRepository
    from marvis.orchestrator.reconciliation import register_governed_outcome_verifier

    case = _case(tmp_path)
    if applied:
        _adopt(case)
    else:
        _reserve(case)
    governance = GovernanceRepository(
        case.settings.db_path, runtime_generation="after-cancel"
    )
    governance.reconcile_startup()
    case.plans.finish_step_run(
        case.run_id,
        status="interrupted",
        error_kind="unknown_effect",
        error="cancelled",
    )
    plan = case.plans.load_plan("producer-plan")
    step = plan.steps[0]
    step.status, step.error = (
        StepStatus.FAILED,
        "cancelled after dispatch; explicit reconciliation required",
    )
    case.plans.update_step(step)
    case.plans.set_plan_status(plan.id, PlanStatus.CANCELLED)
    effects, hooks, runner = [], _hooks(case.plans), FakeRunner([])
    hooks.register_listener(
        "step.completed", lambda event, payload: effects.append(payload), required=True
    )
    executor = _executor(case.plans, runner, hooks=hooks, reviewer=CountingReviewer())
    register_governed_outcome_verifier(executor.reconciler.verifiers, governance)
    client = _client_for(case.plans, executor)
    target = client.get("/api/plans/producer-plan").json()["plan"]["reconciliation"][
        "targets"
    ][0]
    assert target["action"] == "reconcile"
    assert (
        client.post(
            "/api/plans/producer-plan/resume-completion",
            json={"target_id": target["id"]},
        ).status_code
        == 409
    )
    result = client.post(
        "/api/plans/producer-plan/reconcile", json={"target_id": target["id"]}
    )
    assert result.status_code == 200, result.text
    assert result.json()["plan"]["status"] == "cancelled"
    assert effects == [] and runner.calls == []
    if applied:
        with pytest.raises(ConflictError):
            case.plans.retry_failed_step(plan.id, step.id)
        completion = result.json()["plan"]["reconciliation"]["targets"][0]
        assert completion["action"] == "resume_completion"
        response = client.post(
            "/api/plans/producer-plan/resume-completion",
            json={"target_id": completion["id"]},
        )
        assert response.status_code == 200, response.text
        assert response.json()["plan"]["status"] == "done"
        assert len(effects) == 1 and runner.calls == []
    else:
        assert result.json()["reconciliation_result"]["outcome"] == "not_applied_fenced"
        assert case.plans.retry_failed_step(plan.id, step.id) == [step.id]
        assert not case.plans.is_step_confirmed(step.id)
        assert governance.get_approval(case.grant.approval.id).state.value == "revoked"
