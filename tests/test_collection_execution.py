"""Real native reservations, governed producer receipts and restart readback."""

from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace
from uuid import uuid4

import pytest

from marvis.collection.actions import CollectionPolicy
from marvis.collection.batches import CollectionBatchRequest
from marvis.collection.execution import CollectionExecutor
from marvis.collection.execution_state import batch_on_connection
from marvis.collection.ledger import CollectionEvidenceError
from marvis.db import PluginRepository, connect
from marvis.governance.repository import GovernanceRepository
from marvis.governance.service import GovernanceService
from marvis.orchestrator.contracts import Plan, PlanStatus, PlanStep, StepStatus
from marvis.packs.collection.tools import _execute
from marvis.plugins.contracts import ToolContext
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.manifest import ToolRef
from marvis.plugins.registry import PluginRegistry, ToolRegistry
from marvis.plugins.runner import ToolRunner
from marvis.repositories.plans import PlanRepository
from marvis.repositories.strategy import StrategyRepository
from marvis.state_machine import ConflictError
from tests.test_collection_batches import batch as batch
from tests.test_collection_cashflows import ledger as ledger
from tests.test_collection_planning import spec


def executable(request, **changes):
    body = request.model_dump()
    p = body["policy"]
    p["max_estimated_active_cost_minor"] = 1000
    for q in p["queues"]:
        q["max_active_actions"] = 10
    body.update(changes)
    body["strategy"] = spec(CollectionPolicy.model_validate(body["policy"])).to_dict()
    return CollectionBatchRequest.model_validate(body)


def governed(batch, operation="queue_batch", *, request=None, actor=None):
    store, task, original, actors = batch
    request = request or executable(original)
    prepared = store.prepare(task, request, actors["maker"])
    CollectionExecutor(store.settings)
    plugins = PluginRepository(store.settings.db_path)
    registry = PluginRegistry(plugins)
    load_builtin_packs(registry, Path(__file__).parents[1] / "marvis" / "packs")
    tools = ToolRegistry(registry)
    plans = PlanRepository(store.settings.db_path)
    governance = GovernanceRepository(
        store.settings.db_path, runtime_generation="collection-runtime"
    )
    service = GovernanceService(
        plan_repo=plans,
        tool_registry=tools,
        strategy_repo=StrategyRepository(store.settings.db_path),
        governance_repo=governance,
    )
    inputs = {
        key: prepared[key] for key in ("batch_id", "request_hash", "preview_hash")
    }
    ref = ToolRef("collection", operation)
    manifest, tool = tools.resolve_with_manifest(ref)
    pid, sid = uuid4().hex, uuid4().hex
    plan = Plan(
        id=pid,
        task_id=task,
        goal=operation,
        source="template",
        template_id="collection_" + operation,
        autonomy_level=1,
        status=PlanStatus.AWAITING_CONFIRM,
        steps=[
            PlanStep(
                id=sid,
                plan_id=pid,
                index=0,
                title=operation,
                tool_ref=ref,
                inputs=inputs,
                depends_on=[],
                post_checks=[],
                needs_confirmation=True,
                policy=tool.policy,
                status=StepStatus.AWAITING_CONFIRM,
            )
        ],
    )
    plans.create_plan(plan)
    grant = service.authorize_step(
        plan_id=pid,
        step_id=sid,
        principal=governance.get_local_principal(actor or actors["checker"]),
        reason="Reviewed explicit local reference proposal",
        expected_plan_revision=0,
    )
    current = plans.load_plan(pid)
    plans.update_step(replace(current.steps[0], status=StepStatus.RUNNING))
    plans.set_plan_status(pid, PlanStatus.RUNNING)
    runner = ToolRunner(
        tools,
        plugins,
        python_executable=sys.executable,
        datasets_root=store.settings.datasets_dir,
        workspace=store.settings.workspace,
        governance=governance,
        binding_resolver=service,
    )
    contract = runner.prepare_invocation(ref)
    runid = plans.start_step_run(
        plan_id=pid,
        step_id=sid,
        tool_ref=ref.label(),
        inputs=inputs,
        invocation_contract=contract,
    )
    binding = service.resolve_binding(
        task_id=task,
        ref=ref,
        inputs=inputs,
        execution_context=grant.context,
        manifest=manifest,
        tool=tool,
    )
    ctx = ToolContext(
        task_id=task,
        seed=42,
        datasets_root=store.settings.datasets_dir,
        workspace=store.settings.workspace,
    )
    return SimpleNamespace(
        store=store,
        settings=store.settings,
        task=task,
        request=request,
        actors=actors,
        plans=plans,
        governance=governance,
        service=service,
        runner=runner,
        inputs=inputs,
        ref=ref,
        grant=grant,
        contract=contract,
        runid=runid,
        binding=binding,
        ctx=ctx,
    )


def invoke(c):
    return c.runner.invoke(
        c.ref,
        c.inputs,
        task_id=c.task,
        execution_context=c.grant.context,
        invocation_id=c.runid,
        expected_invocation=c.contract,
        on_dispatch=lambda: c.plans.mark_step_run_dispatched(c.runid),
    )


def reserve(c):
    c.plans.mark_step_run_dispatched(c.runid)
    e = c.governance.reserve_effect(
        c.grant.context,
        c.binding,
        invocation_id=c.runid,
        invocation_contract=c.contract,
    )
    c.governance.mark_effect_dispatched(e.id, reservation_id=e.reservation_id)
    c.effect = e
    c.ctx = replace(
        c.ctx, effect_execution_id=e.id, runtime_generation=e.runtime_generation
    )
    return c


def apply(c):
    reserve(c)
    return _execute(c.inputs, c.ctx, c.ref.tool)


def test_original_policy_wire_has_no_implicit_execution_defaults(batch):
    _, _, request, _ = batch
    payload = request.model_dump()
    assert "max_estimated_active_cost_minor" not in payload["policy"]
    assert all("max_active_actions" not in q for q in payload["policy"]["queues"])
    p = {**payload["policy"], "max_estimated_active_cost_minor": None}
    p["queues"] = [{**q, "max_active_actions": None} for q in p["queues"]]
    assert CollectionPolicy.model_validate(p).model_dump() == payload["policy"]
    c = governed(batch, request=request)
    with pytest.raises(ValueError, match="capacity_or_budget_unknown"):
        apply(c)
    assert (
        c.store.read(c.task, c.request.batch_id, c.actors["maker"])["status"]
        == "proposed"
    )


def test_real_worker_queue_execute_and_original_receipt_recovery(batch):
    c = governed(batch)
    result = invoke(c)
    assert result.ok, result
    output = result.output
    assert output["status"] == "queued" and output["action_count"] == 1
    assert (
        output["actual_cost_minor"] is None
        and output["external_action_executed"] is False
    )
    recovered = c.governance.verify_producer_outcome(c.runid)
    assert recovered["outcome"] == "applied", recovered
    assert recovered["output"] == output
    done = governed(batch, "execute_reference", request=c.request)
    result = invoke(done)
    assert result.ok, result
    assert result.output["status"] == "completed"
    assert c.governance.verify_producer_outcome(c.runid)["output"] == output
    with connect(c.settings.db_path) as conn:
        item = json.loads(
            conn.execute("SELECT payload_json FROM collection_queue_items").fetchone()[
                0
            ]
        )
        final = json.loads(
            conn.execute(
                "SELECT payload_json FROM collection_action_finals"
            ).fetchone()[0]
        )
        assert item["assigned_to"] == c.actors["maker"]
        assert final["customer_contacted"] is False
        assert (
            conn.execute(
                "SELECT count(*) FROM collection_evidence WHERE kind='flow'"
            ).fetchone()[0]
            == 0
        )


def test_cancellation_is_governed_and_releases_reservation(batch):
    c = governed(batch)
    assert apply(c)["action_count"] == 1
    cancel = governed(batch, "cancel_batch", request=c.request)
    output = apply(cancel)
    assert output["status"] == "cancelled"
    assert (
        cancel.governance.verify_producer_outcome(cancel.runid)["outcome"] == "applied"
    )
    second = governed(batch, request=executable(batch[2], batch_id="second"))
    assert apply(second)["action_count"] == 1


def test_mutable_batch_head_cannot_forge_producer_history(batch):
    c = governed(batch)
    apply(c)
    with connect(c.settings.db_path) as conn:
        conn.execute("UPDATE collection_batches SET status='proposed',revision=1")
    with pytest.raises(CollectionEvidenceError, match="head_integrity"):
        c.store.read(c.task, c.request.batch_id, c.actors["maker"])


def test_live_subject_frequency_holds_later_batch_and_does_not_erase_pending(batch):
    first = governed(batch)
    assert apply(first)["action_count"] == 1
    second = governed(batch, request=executable(batch[2], batch_id="second"))
    output = apply(second)
    assert output["action_count"] == 0 and output["held_count"] == 1
    with connect(first.settings.db_path) as conn:
        assert (
            conn.execute("SELECT count(*) FROM collection_queue_items").fetchone()[0]
            == 1
        )
        assert (
            conn.execute("SELECT count(*) FROM collection_action_finals").fetchone()[0]
            == 0
        )


def test_writer_rechecks_role_source_and_approval(batch):
    c = reserve(governed(batch))
    (c.settings.tasks_dir / c.task / "source.json").write_text("changed")
    with pytest.raises(ValueError, match="source_integrity"):
        _execute(c.inputs, c.ctx, "queue_batch")
    with connect(c.settings.db_path) as conn:
        assert (
            conn.execute("SELECT count(*) FROM collection_queue_items").fetchone()[0]
            == 0
        )


def test_native_rows_immutable_and_readback_read_only(batch):
    c = governed(batch)
    output = apply(c)
    with connect(c.settings.db_path) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute("DELETE FROM collection_queue_items")
        assert (
            batch_on_connection(
                conn, c.task, c.request.batch_id, c.store.ledger.secret
            )["status"]
            == "queued"
        )
    assert c.governance.verify_producer_outcome(c.runid)["output"] == output


@pytest.mark.parametrize(
    "failure",
    [
        "maker_revoked",
        "checker_revoked",
        "approval_revoked",
        "expired",
        "stale_revision",
    ],
)
def test_writer_revalidates_live_authority_before_native_reservation(batch, failure):
    c = reserve(governed(batch))
    with connect(c.settings.db_path) as conn:
        if failure in {"maker_revoked", "checker_revoked"}:
            actor = c.actors["maker" if failure == "maker_revoked" else "checker"]
            conn.execute(
                "UPDATE production_principals SET status='revoked' WHERE local_principal_id=?",
                (actor,),
            )
        elif failure == "approval_revoked":
            conn.execute(
                "UPDATE approval_records SET status='revoked' WHERE id=?",
                (c.effect.approval_id,),
            )
        elif failure == "expired":
            conn.execute(
                "UPDATE approval_records SET expires_at='2020-01-01T00:00:00Z' WHERE id=?",
                (c.effect.approval_id,),
            )
        else:
            conn.execute("UPDATE collection_batches SET revision=revision+1")
    with pytest.raises((ValueError, RuntimeError)):
        _execute(c.inputs, c.ctx, "queue_batch")
    with connect(c.settings.db_path) as conn:
        assert (
            conn.execute("SELECT count(*) FROM collection_queue_items").fetchone()[0]
            == 0
        )
        assert (
            conn.execute("SELECT count(*) FROM collection_native_effects").fetchone()[0]
            == 0
        )


def test_direct_toolrunner_cannot_bypass_human_and_effect_gates(batch):
    c = governed(batch)
    result = c.runner.invoke(c.ref, c.inputs, task_id=c.task)
    assert not result.ok
    with connect(c.settings.db_path) as conn:
        assert (
            conn.execute("SELECT count(*) FROM collection_queue_items").fetchone()[0]
            == 0
        )


def another_task(
    batch, *, token="a" * 64, namespace="fixture", capacity=10, budget=1000
):
    from marvis.collection.batches import CollectionBatchStore
    from marvis.collection.contracts import CollectionCase
    from marvis.repositories.task_artifacts import TaskArtifactRepository
    from marvis.repositories.tasks import TaskRepository
    from tests.test_db import _task_create

    store, task, request, actors = batch
    other = (
        TaskRepository(store.settings.db_path)
        .create_task(_task_create(task_type="strategy"))
        .id
    )
    source = store.settings.tasks_dir / other / "source.json"
    source.parent.mkdir(parents=True)
    source.write_bytes((store.settings.tasks_dir / task / "source.json").read_bytes())
    a = TaskArtifactRepository(store.settings.db_path).register(
        task_id=other,
        kind="collection_source",
        path=str(source.relative_to(store.settings.workspace)),
        content_hash=request.policy.basis_artifact_hash,
        origin_tool="synthetic_test",
        provenance={},
    )
    evidence = {
        "source_artifact_id": a["id"],
        "source_artifact_hash": a["content_hash"],
    }
    store.ledger.create_case(
        other,
        CollectionCase(
            case_id="case-1",
            subject_namespace=namespace,
            subject_token=token,
            unit=request.policy.unit,
            opening_balance_minor=10000,
            opened_at="2026-08-01T00:00:00Z",
            source_assurance="historical_import_unverified",
            **evidence,
        ),
    )
    p = executable(request).model_dump()
    p["policy"].update(
        basis_artifact_id=a["id"],
        basis_artifact_hash=a["content_hash"],
        max_estimated_active_cost_minor=budget,
    )
    for q in p["policy"]["queues"]:
        q["max_active_actions"] = capacity
    for item in p["cases"] + p["histories"]:
        item.update(subject_namespace=namespace, subject_token=token, **evidence)
    p["strategy"] = spec(CollectionPolicy.model_validate(p["policy"])).to_dict()
    return (
        CollectionBatchStore(store.settings),
        other,
        CollectionBatchRequest.model_validate(p),
        actors,
    )


def test_same_subject_cross_task_reservation_is_counted(batch):
    first = governed(batch)
    apply(first)
    foreign = another_task(batch)
    second = governed(foreign, request=foreign[2])
    assert apply(second)["action_count"] == 0


@pytest.mark.parametrize("limit", ["capacity", "budget", "namespace"])
def test_queue_resource_scope_crosses_tasks_and_policy_revisions(batch, limit):
    first = governed(batch)
    apply(first)
    foreign = another_task(
        batch,
        token="b" * 64 if limit != "namespace" else "a" * 64,
        namespace="other",
        capacity=1 if limit == "capacity" else 10,
        budget=0 if limit == "budget" else 1000,
    )
    payload = foreign[2].model_dump()
    payload["policy"]["revision"] = "2"
    payload["strategy"] = spec(
        CollectionPolicy.model_validate(payload["policy"])
    ).to_dict()
    second = governed(foreign, request=CollectionBatchRequest.model_validate(payload))
    output = apply(second)
    assert output["action_count"] == (1 if limit == "namespace" else 0)


def test_concurrent_reservation_is_serialized_by_original_writer(batch):
    from concurrent.futures import ThreadPoolExecutor

    first = reserve(governed(batch))
    second = reserve(governed(batch, request=executable(batch[2], batch_id="second")))
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(
            pool.map(
                lambda c: _execute(c.inputs, c.ctx, "queue_batch"), [first, second]
            )
        )
    assert sorted(v["action_count"] for v in outcomes) == [0, 1]


def test_lost_host_return_recovers_original_worker_commit_without_reexecution(
    batch, monkeypatch
):
    class HostCrash(BaseException):
        pass

    c = governed(batch)
    original = c.runner._finalize_effect_result

    def crash(*args, **kwargs):
        result = original(*args, **kwargs)
        if result.ok:
            raise HostCrash("after real worker commit")
        return result

    monkeypatch.setattr(c.runner, "_finalize_effect_result", crash)
    with pytest.raises(HostCrash):
        invoke(c)
    recovered = GovernanceRepository(
        c.settings.db_path, runtime_generation="new-runtime"
    )
    recovered.reconcile_startup()
    outcome = recovered.verify_producer_outcome(c.runid)
    assert outcome["outcome"] == "applied", outcome
    assert outcome["output"]["action_count"] == 1
    with connect(c.settings.db_path) as conn:
        assert (
            conn.execute("SELECT count(*) FROM collection_queue_items").fetchone()[0]
            == 1
        )
        assert (
            conn.execute("SELECT count(*) FROM collection_native_effects").fetchone()[0]
            == 1
        )
    assert recovered.verify_producer_outcome(c.runid) == outcome


def test_explicit_native_history_identity_deduplicates_only_exact_attempt(batch):
    from marvis.collection.contracts import CollectionCase
    from tests.test_collection_planning import attempt

    first = governed(batch)
    apply(first)
    with connect(first.settings.db_path) as conn:
        native = json.loads(
            conn.execute("SELECT payload_json FROM collection_queue_items").fetchone()[
                0
            ]
        )
    # Count limit two and no minimum interval: one native reservation, one new case.
    payload = executable(batch[2], batch_id="second").model_dump()
    payload["policy"]["min_contact_interval_seconds"] = 0
    payload["cases"][0]["case_id"] = "case-2"
    old = first.store.ledger._get
    with connect(first.settings.db_path) as conn:
        case = CollectionCase.model_validate(
            {**old(conn, first.task, "case", "case-1"), "case_id": "case-2"}
        )
    first.store.ledger.create_case(first.task, case)
    payload["histories"][0]["attempts"] = [
        attempt(
            payload["as_of"],
            state="reserved",
            case_id="case-1",
            reference_action_id=native["id"],
        )
    ]
    payload["strategy"] = spec(
        CollectionPolicy.model_validate(payload["policy"])
    ).to_dict()
    second = governed(batch, request=CollectionBatchRequest.model_validate(payload))
    assert apply(second)["action_count"] == 1
    payload["batch_id"] = "third"
    payload["histories"][0]["attempts"][0]["state"] = "cancelled_before_dispatch"
    bad = governed(batch, request=CollectionBatchRequest.model_validate(payload))
    with pytest.raises(ValueError, match="source_conflict"):
        apply(bad)


def test_tampered_native_file_cannot_be_reconciled_as_applied(batch):
    c = governed(batch)
    apply(c)
    path = next((c.settings.tasks_dir / c.task / "collection").glob("execution-*.json"))
    path.write_text("{}")
    assert c.governance.verify_producer_outcome(c.runid)["outcome"] == "unknown"


def test_unrelated_maker_cannot_apply_another_makers_batch(batch):
    c = governed(batch, actor=batch[3]["other"])
    with pytest.raises(ValueError, match="approver_scope_forbidden"):
        apply(c)


def test_checker_can_cancel_after_maker_revocation_and_source_drift(batch):
    c = governed(batch)
    apply(c)
    # Prepare/approve cancellation while owner is active, then revoke the owner;
    # cleanup is authorized by the still-active checker, independent of drift.
    cancellation = governed(batch, "cancel_batch", request=c.request)
    with connect(c.settings.db_path) as conn:
        conn.execute(
            "UPDATE production_principals SET status='revoked' WHERE local_principal_id=?",
            (c.actors["maker"],),
        )
    (c.settings.tasks_dir / c.task / "source.json").write_text("changed")
    assert apply(cancellation)["status"] == "cancelled"


def test_review_queue_keeps_unestimated_cost_unknown(batch):
    from tests.test_collection_planning import action

    request = executable(batch[2])
    # Typed review action cannot declare a contact cost; its estimate is unknown.
    payload = request.model_dump()
    payload["strategy"] = spec(
        request.policy,
        default_action=action(
            request.policy, kind="review", channel=None, estimated_cost_minor=None
        ),
    ).to_dict()
    c = governed(batch, request=CollectionBatchRequest.model_validate(payload))
    result = invoke(c)
    assert result.ok, result
    assert result.output["action_count"] == 1
    assert result.output["unestimated_review_count"] == 1
    assert result.output["estimated_cost_scope"] == "contact_actions_only"
    with connect(c.settings.db_path) as conn:
        item = json.loads(
            conn.execute("SELECT payload_json FROM collection_queue_items").fetchone()[
                0
            ]
        )
        assert item["estimated_cost_minor"] is None


@pytest.mark.parametrize("checkpoint", ["before_native_verify", "after_native_verify"])
def test_atomic_producer_failure_rolls_back_then_fences_late_worker(
    batch, monkeypatch, checkpoint
):
    import marvis.collection.execution as execution

    c = reserve(governed(batch))
    original = execution.verify_output

    def fail(*args, **kwargs):
        if checkpoint == "after_native_verify":
            original(*args, **kwargs)
        raise RuntimeError("producer checkpoint failure")

    with monkeypatch.context() as patch:
        patch.setattr(execution, "verify_output", fail)
        with pytest.raises(RuntimeError, match="checkpoint"):
            _execute(c.inputs, c.ctx, "queue_batch")
    with connect(c.settings.db_path) as conn:
        assert (
            conn.execute("SELECT count(*) FROM collection_queue_items").fetchone()[0]
            == 0
        )
        assert (
            conn.execute("SELECT count(*) FROM collection_native_effects").fetchone()[0]
            == 0
        )
        assert (
            conn.execute(
                "SELECT count(*) FROM task_artifacts WHERE kind='collection_reference_execution'"
            ).fetchone()[0]
            == 0
        )
    assert not list(
        (c.settings.tasks_dir / c.task / "collection").glob("execution-*.json")
    )
    assert c.governance.verify_producer_outcome(c.runid)["outcome"] == "unknown"
    restarted = GovernanceRepository(
        c.settings.db_path, runtime_generation="after-checkpoint-failure"
    )
    restarted.reconcile_startup()
    assert restarted.verify_producer_outcome(c.runid)["outcome"] == "not_applied_fenced"
    with pytest.raises(ValueError, match="not_committable"):
        _execute(c.inputs, c.ctx, "queue_batch")


def test_original_collection_receipt_verifier_is_read_only(batch):
    c = governed(batch)
    output = apply(c)
    with connect(c.settings.db_path) as conn:
        denied = {sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE}
        conn.set_authorizer(
            lambda op, *args: sqlite3.SQLITE_DENY if op in denied else sqlite3.SQLITE_OK
        )
        outcome = c.governance.verify_producer_outcome(c.runid, conn=conn)
    assert outcome["outcome"] == "applied"
    assert outcome["output"] == output


@pytest.mark.parametrize("committed", [True, False])
def test_existing_http_reconciler_restores_original_collection_or_fenced_retry(
    batch, committed
):
    from marvis.orchestrator.reconciliation import register_governed_outcome_verifier
    from tests.test_trusted_reconciliation import _client_for
    from tests.test_orch_completion import CountingReviewer, _hooks
    from tests.test_orch_executor import FakeRunner, _executor

    c = governed(batch)
    if committed:
        result = invoke(c)
        assert result.ok, result
        output = result.output
    else:
        reserve(c)
    restarted = GovernanceRepository(
        c.settings.db_path, runtime_generation="collection-restart"
    )
    restarted.reconcile_startup()
    c.plans.finish_step_run(
        c.runid,
        status="interrupted",
        error_kind="unknown_effect",
        error="host lost worker return",
    )
    plan = c.plans.load_plan(c.binding.plan_id)
    step = plan.steps[0]
    step.status, step.error = (
        StepStatus.FAILED,
        "unknown effect requires original producer proof",
    )
    c.plans.update_step(step)
    c.plans.set_plan_status(plan.id, PlanStatus.FAILED)
    runner = FakeRunner([])
    executor = _executor(
        c.plans, runner, hooks=_hooks(c.plans), reviewer=CountingReviewer()
    )
    register_governed_outcome_verifier(executor.reconciler.verifiers, restarted)
    client = _client_for(c.plans, executor)
    target = client.get("/api/plans/" + plan.id).json()["plan"]["reconciliation"][
        "targets"
    ][0]
    assert target["supported"] is True
    response = client.post(
        "/api/plans/" + plan.id + "/reconcile", json={"target_id": target["id"]}
    )
    assert response.status_code == 200, response.text
    if committed:
        assert response.json()["plan"]["status"] == "done"
        assert c.plans.load_step_output(step.id) == output
        assert c.plans.list_step_runs(step.id)[0]["status"] == "succeeded"
    else:
        assert (
            c.plans.latest_failed_step_run_error_kind(step.id)
            == "execution_not_applied"
        )
        assert c.plans.retry_failed_step(plan.id, step.id) == [step.id]
        assert not c.plans.is_step_confirmed(step.id)
        assert restarted.get_approval(c.grant.approval.id).state.value == "revoked"
    assert runner.calls == []


@pytest.mark.parametrize("state", ["queued", "completed", "cancelled"])
def test_reference_reservations_survive_attempted_task_deletion(batch, state):
    from marvis.repositories.tasks import TaskRepository

    c = governed(batch)
    apply(c)
    if state != "queued":
        apply(
            governed(
                batch,
                "execute_reference" if state == "completed" else "cancel_batch",
                request=c.request,
            )
        )
    repo = TaskRepository(c.settings.db_path)
    before = c.governance.verify_producer_outcome(c.runid)
    # Both normal repository paths and direct SQL deletion must fail atomically.
    for action in (lambda: repo.delete_task(c.task), lambda: repo.purge_task(c.task)):
        with pytest.raises(
            (sqlite3.IntegrityError, ConflictError),
            match="collection_reference_evidence_retained",
        ):
            action()
    with connect(c.settings.db_path) as conn:
        with pytest.raises(
            sqlite3.IntegrityError, match="collection_reference_evidence_retained"
        ):
            conn.execute("DELETE FROM tasks WHERE id=?", (c.task,))
        assert (
            conn.execute("SELECT count(*) FROM collection_queue_items").fetchone()[0]
            == 1
        )
        assert (
            conn.execute(
                "SELECT count(*) FROM task_artifacts WHERE task_id=?", (c.task,)
            ).fetchone()[0]
            >= 2
        )
    assert c.governance.verify_producer_outcome(c.runid) == before
    other = another_task(batch)
    second = governed(other, request=other[2])
    assert apply(second)["action_count"] == (1 if state == "cancelled" else 0)


def test_proposal_without_reference_reservations_remains_deletable(batch):
    from marvis.repositories.tasks import TaskRepository

    store, task, request, actors = batch
    CollectionExecutor(store.settings)
    store.prepare(task, request, actors["maker"])
    TaskRepository(store.settings.db_path).delete_task(task)
    with connect(store.settings.db_path) as conn:
        assert (
            conn.execute("SELECT count(*) FROM collection_batches").fetchone()[0] == 0
        )


def test_failed_purge_cannot_release_capacity_for_another_subject_task(batch):
    from marvis.repositories.tasks import TaskRepository

    c = governed(batch)
    apply(c)
    with pytest.raises(ConflictError, match="collection_reference_evidence_retained"):
        TaskRepository(c.settings.db_path).purge_task(c.task)
    other = another_task(batch, token="b" * 64, namespace="independent", capacity=1)
    second = governed(other, request=other[2])
    output = apply(second)
    assert output["action_count"] == 0
    with connect(c.settings.db_path) as conn:
        receipt = json.loads(
            conn.execute(
                "SELECT payload_json FROM collection_native_effects WHERE invocation_id=?",
                (second.runid,),
            ).fetchone()[0]
        )
    assert receipt["facts"]["held"] == [
        {"case_id": "case-1", "reason": "live_queue_capacity"}
    ]
