"""Real native workers lose their host return, then restore through completion only."""

from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest
import pandas as pd

from marvis.app import create_app
from marvis.db_schema import connect
from marvis.governance.native_producers import KIND, _path
from marvis.orchestrator.evidence import payload_hash
from test_event_runtime import runtime, prepare, gated_plan, approve
import test_decision_twin_event_batch as batches
from test_decision_twin_batch import batch as ordinary_batch  # noqa: F401
from test_reference_decision import packaged  # noqa: F401


def _lose_return(app, monkeypatch, tool_ref, on_commit=None, *, fail=True):
    original = app.state.tool_runner._finalize_effect_result
    observed = {"calls": 0}

    def lose(*args, **kwargs):
        result = original(*args, **kwargs)
        if args[1].split("@", 1)[0] == tool_ref and result.ok:
            observed["calls"] += 1
            observed["output"] = result.output
            if on_commit is not None:
                on_commit()
            if fail:
                raise RuntimeError(
                    "synthetic host return lost after actual worker commit"
                )
        return result

    monkeypatch.setattr(app.state.tool_runner, "_finalize_effect_result", lose)
    return observed


def _case(
    tmp_path,
    monkeypatch,
    kind="event",
    *,
    before_dispatch=False,
    unbound=False,
    cancelled=False,
    completion_loss=False,
):
    rt = runtime.__wrapped__(tmp_path)
    if kind == "reconciliation":
        rt.anchor -= timedelta(days=120)
    if kind == "event":
        proposed = prepare(rt)
        plan = gated_plan(rt, proposed)
        tool_ref = "risk_context.replay_events"

        def run():
            return approve(rt, plan)

        task_id = rt.task.id
    else:
        ctx = batches.batch(rt)
        proposed = batches.proposal(ctx)
        tool_ref = "decision_twin.replay_history"
        goal, slot = "历史决策回放", "replay_contract"
        if kind == "reconciliation":
            original = batches.replay(ctx)
            path = rt.root / "outcomes.parquet"
            pd.DataFrame(
                {
                    "id": ctx.frame["id"],
                    "observed_at": datetime.now(UTC).isoformat(),
                    "loss": [3.0, 4.0],
                    "profit": [20.0, 30.0],
                }
            ).to_parquet(path, index=False)
            record = ctx.material.registry.register_existing(
                path, task_id=ctx.material.task_id, role="historical_outcomes"
            )
            contract = {
                "replay_artifact_id": original["artifact_id"],
                "dataset_id": record.id,
                "expected_content_hash": record.content_hash,
                "record_id_col": "id",
                "observed_at_col": "observed_at",
                "actual_loss_col": "loss",
                "actual_profit_col": "profit",
                "currency": "CNY",
                "maturity_days": 90,
                "maturity_source_ref": "90 day source",
                "source_ref": "external history",
                "reconciled_at": datetime.now(UTC).isoformat(),
            }
            response = rt.maker.post(
                f"/api/tasks/{ctx.material.task_id}/decision-twin/reconciliation-proposal",
                json=contract,
            )
            assert response.status_code == 200, response.text
            proposed = response.json()
            tool_ref, goal, slot = (
                "decision_twin.reconcile_history",
                "历史决策现金流对账",
                "reconciliation_contract",
            )

        def run():
            return batches.run(ctx, proposed, goal=goal, slot=slot)

        task_id = ctx.material.task_id

    def cancel_after_commit():
        response = rt.maker.post(f"/api/plans/{plan['id']}/cancel")
        assert response.status_code == 200, response.text

    observed = _lose_return(
        rt.app,
        monkeypatch,
        tool_ref,
        cancel_after_commit if cancelled else None,
        fail=not completion_loss,
    )
    if completion_loss:

        def lose_completion(*args):
            raise RuntimeError("synthetic interrupted completion after durable output")

        monkeypatch.setattr(
            rt.app.state.plan_executor._reviewer, "deterministic_check", lose_completion
        )
    if before_dispatch or unbound:
        import marvis.plugins.runner as module

        original = module._run_worker

        def changed(python, job, **kwargs):
            if (job["module"], job["entrypoint"]) in {
                ("marvis.packs.risk_context.tools", "tool_replay_events"),
                ("marvis.packs.decision_twin.tools", "tool_replay_history"),
                ("marvis.packs.decision_twin.tools", "tool_reconcile_history"),
            }:
                if before_dispatch:
                    raise subprocess.TimeoutExpired("synthetic never-started-worker", 1)
                job = {k: v for k, v in job.items() if k != "invocation_id"}
            return original(python, job, **kwargs)

        monkeypatch.setattr(module, "_run_worker", changed)
    failed = run()
    assert failed["status"] == ("cancelled" if cancelled else "failed"), failed
    target = next(
        t
        for t in failed["reconciliation"]["targets"]
        if t["kind"] == ("completion" if completion_loss else "tool")
    )
    assert target["supported"] is True
    step = failed["steps"][0]
    runs = rt.app.state.plan_repo.list_step_runs(step["id"])
    assert len(runs) == 1
    assert (runs[0]["output_ref"] is not None) == completion_loss
    return SimpleNamespace(
        rt=rt,
        task_id=task_id,
        plan=failed,
        target=target,
        run=runs[0],
        observed=observed,
        tool_ref=tool_ref,
    )


def _fresh(case, monkeypatch):
    app = create_app(case.rt.settings)
    maker = TestClient(app)
    maker.cookies.update(case.rt.maker.cookies)

    def never(*args, **kwargs):
        raise AssertionError("recovery must not run any Tool")

    monkeypatch.setattr(app.state.tool_runner, "invoke", never)
    monkeypatch.setattr(app.state.risk_events, "execute", never)
    monkeypatch.setattr(app.state.risk_sources, "execute", never)
    return app, maker


def _records(case):
    with connect(case.rt.settings.db_path) as conn:
        return [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM task_artifacts WHERE task_id=? AND kind=?",
                (case.task_id, KIND),
            ).fetchall()
        ]


@pytest.mark.parametrize("kind", ["event", "batch", "reconciliation"])
def test_real_commit_host_loss_restart_restores_exact_output_without_rerun(
    tmp_path, monkeypatch, kind
):
    case = _case(tmp_path, monkeypatch, kind)
    assert case.observed["calls"] == 1 and len(_records(case)) == 1
    native = json.loads(Path(_records(case)[0]["path"]).read_text())["body"]
    assert native["binding"]["invocation_id"] == case.run["id"]
    assert native["binding"]["invocation_contract"] == case.run["invocation_contract"]
    assert native["binding"]["input_hash"] == payload_hash(case.run["input"])
    assert native["output"] == case.observed["output"]
    app, maker = _fresh(case, monkeypatch)
    response = maker.post(
        f"/api/plans/{case.plan['id']}/reconcile", json={"target_id": case.target["id"]}
    )
    assert response.status_code == 200, response.text
    assert response.json()["reconciliation_result"]["outcome"] == "applied"
    assert response.json()["plan"]["status"] == "done", response.json()
    assert (
        app.state.plan_repo.load_step_output(case.plan["steps"][0]["id"])
        == case.observed["output"]
    )
    assert len(app.state.plan_repo.list_step_runs(case.plan["steps"][0]["id"])) == 1
    repeated = maker.post(
        f"/api/plans/{case.plan['id']}/reconcile", json={"target_id": case.target["id"]}
    )
    assert repeated.status_code == 200, repeated.text
    assert len(_records(case)) == 1


@pytest.mark.parametrize("unbound", [False, True])
def test_missing_receipt_is_unknown_even_if_identical_domain_artifact_exists(
    tmp_path, monkeypatch, unbound
):
    case = _case(tmp_path, monkeypatch, before_dispatch=not unbound, unbound=unbound)
    assert _records(case) == []
    with connect(case.rt.settings.db_path) as conn:
        count = conn.execute(
            "SELECT count(*) FROM task_artifacts WHERE task_id=? AND kind='risk_event_features'",
            (case.task_id,),
        ).fetchone()[0]
    assert count == int(unbound)
    app, maker = _fresh(case, monkeypatch)
    response = maker.post(
        f"/api/plans/{case.plan['id']}/reconcile", json={"target_id": case.target["id"]}
    )
    assert response.status_code == 200, response.text
    assert response.json()["reconciliation_result"]["outcome"] == "unknown"
    assert response.json()["plan"]["status"] == "failed"
    assert (
        app.state.plan_repo.list_step_runs(case.plan["steps"][0]["id"])[0]["output_ref"]
        is None
    )


def test_reconciliation_reads_only_original_proof_and_preserves_query_only(
    tmp_path, monkeypatch
):
    case = _case(tmp_path, monkeypatch)
    reconciler = case.rt.app.state.plan_executor.reconciler
    target = reconciler._targets(
        case.rt.app.state.plan_repo.load_plan(case.plan["id"])
    )[0]
    with connect(case.rt.settings.db_path) as conn:
        conn.execute("PRAGMA query_only=ON")
        statements = []
        conn.set_trace_callback(statements.append)
        with pytest.raises(PermissionError, match="来源读取授权"):
            reconciler.verifiers.verify(target, conn)
        assert conn.execute("PRAGMA query_only").fetchone()[0] == 1
        with reconciler.verifiers.reader(case.rt.principal["id"]):
            _, proof = reconciler.verifiers.verify(target, conn)
        assert proof.outcome == "applied"
        assert conn.execute("PRAGMA query_only").fetchone()[0] == 1
        assert not any(
            s.lstrip()
            .upper()
            .startswith(("INSERT", "UPDATE", "DELETE", "CREATE", "REPLACE"))
            for s in statements
        )
    assert case.observed["calls"] == 1


@pytest.mark.parametrize("tamper", ["receipt", "domain"])
def test_tampered_producer_or_domain_bytes_never_become_applied(
    tmp_path, monkeypatch, tamper
):
    case = _case(tmp_path, monkeypatch)
    receipt = _records(case)[0]
    body = json.loads(Path(receipt["path"]).read_text())["body"]
    path = Path(receipt["path"] if tamper == "receipt" else body["artifact"]["path"])
    path.write_text("{}")
    _, maker = _fresh(case, monkeypatch)
    response = maker.post(
        f"/api/plans/{case.plan['id']}/reconcile", json={"target_id": case.target["id"]}
    )
    if tamper == "receipt":
        assert response.status_code == 403, response.text
    else:
        assert (
            response.status_code == 200
            and response.json()["reconciliation_result"]["outcome"] == "unknown"
        ), response.text
    assert (
        case.rt.app.state.plan_repo.list_step_runs(case.plan["steps"][0]["id"])[0][
            "output_ref"
        ]
        is None
    )


@pytest.mark.parametrize("kind", ["event", "batch", "reconciliation"])
@pytest.mark.parametrize("action", ["cross_actor", "no_role", "revoke"])
def test_both_recovery_entrypoints_require_current_source_read_authority(
    tmp_path, monkeypatch, action, kind
):
    case = _case(tmp_path, monkeypatch, kind)
    if action == "revoke":
        assert (
            case.rt.admin.post(
                f"/api/tasks/{case.rt.task.id}/risk-events/grants/{case.rt.grant['grant_id']}/revoke"
            ).status_code
            == 200
        )
        client = case.rt.maker
    else:
        client = case.rt.other if action == "cross_actor" else TestClient(case.rt.app)
    for endpoint in ("reconcile", "resume-completion"):
        response = client.post(
            f"/api/plans/{case.plan['id']}/{endpoint}",
            json={"target_id": case.target["id"]},
        )
        assert response.status_code == 403, response.text
    assert (
        case.rt.app.state.plan_repo.list_step_runs(case.plan["steps"][0]["id"])[0][
            "output_ref"
        ]
        is None
    )


@pytest.mark.parametrize("revoked", [False, True])
def test_cancelled_completion_rechecks_reader_after_the_output_was_recovered(
    tmp_path, monkeypatch, revoked
):
    case = _case(tmp_path, monkeypatch, cancelled=True)
    maker = case.rt.maker
    response = maker.post(
        f"/api/plans/{case.plan['id']}/reconcile", json={"target_id": case.target["id"]}
    )
    assert response.status_code == 200, response.text
    assert response.json()["plan"]["status"] == "cancelled"
    targets = response.json()["plan"]["reconciliation"]["targets"]
    target_id = targets[0]["id"] if targets else case.target["id"]
    response = case.rt.other.post(
        f"/api/plans/{case.plan['id']}/resume-completion", json={"target_id": target_id}
    )
    assert response.status_code == 403, response.text
    if not revoked:
        _, fresh_maker = _fresh(case, monkeypatch)
        response = fresh_maker.post(
            f"/api/plans/{case.plan['id']}/resume-completion",
            json={"target_id": target_id},
        )
        assert response.status_code == 200, response.text
        assert response.json()["plan"]["status"] == "done", response.text
        assert case.observed["calls"] == 1
        return
    assert (
        case.rt.admin.post(
            f"/api/tasks/{case.rt.task.id}/risk-events/grants/{case.rt.grant['grant_id']}/revoke"
        ).status_code
        == 200
    )
    response = maker.post(
        f"/api/plans/{case.plan['id']}/resume-completion", json={"target_id": target_id}
    )
    assert response.status_code == 403, response.text
    assert (
        case.rt.app.state.plan_repo.load_plan(case.plan["id"]).status.value
        == "cancelled"
    )


def test_atomic_receipt_filesystem_failure_rolls_back_domain_artifact_registration(
    tmp_path, monkeypatch
):
    rt = runtime.__wrapped__(tmp_path)
    proposed = prepare(rt)
    plan = gated_plan(rt, proposed)
    (rt.settings.workspace / "native_tool_receipts").write_text("synthetic disk fault")
    result = approve(rt, plan)
    assert result["status"] == "failed", result
    with connect(rt.settings.db_path) as conn:
        assert (
            conn.execute(
                "SELECT count(*) FROM task_artifacts WHERE task_id=? AND kind IN ('risk_event_features',?)",
                (rt.task.id, KIND),
            ).fetchone()[0]
            == 0
        )
    target = result["reconciliation"]["targets"][0]
    response = rt.maker.post(
        f"/api/plans/{plan['id']}/reconcile", json={"target_id": target["id"]}
    )
    assert response.status_code == 200, response.text
    assert response.json()["reconciliation_result"]["outcome"] == "unknown"
    # A private path is derived from the invocation, never from user Tool inputs.
    assert _path(rt.settings, rt.task.id, "../../escape").is_relative_to(
        rt.settings.workspace / "native_tool_receipts"
    )


def test_revocation_between_read_and_writer_transaction_cannot_restore_output(
    tmp_path, monkeypatch
):
    case = _case(tmp_path, monkeypatch)
    verifier = case.rt.app.state.plan_executor.reconciler.verifiers
    original = verifier.verify
    calls = []

    def revoke_after_read(target, conn):
        result = original(target, conn)
        calls.append(result)
        if len(calls) == 1:
            response = case.rt.admin.post(
                f"/api/tasks/{case.rt.task.id}/risk-events/grants/{case.rt.grant['grant_id']}/revoke"
            )
            assert response.status_code == 200, response.text
        return result

    monkeypatch.setattr(verifier, "verify", revoke_after_read)
    response = case.rt.maker.post(
        f"/api/plans/{case.plan['id']}/reconcile", json={"target_id": case.target["id"]}
    )
    assert response.status_code == 403, response.text
    assert len(calls) == 1
    assert (
        case.rt.app.state.plan_repo.list_step_runs(case.plan["steps"][0]["id"])[0][
            "output_ref"
        ]
        is None
    )


def test_internal_native_receipt_is_not_a_generic_download(tmp_path, monkeypatch):
    from test_restricted_risk_artifacts import generic_urls

    case = _case(tmp_path, monkeypatch)
    record = _records(case)[0]
    urls = generic_urls(case.rt.settings, case.task_id, record)
    for client in (case.rt.maker, case.rt.other, TestClient(case.rt.app)):
        assert client.get(urls[0]).status_code == 403
        # The path endpoint also rejects the original private location.
        assert client.get(urls[1]).status_code == 404
        listing = client.get(f"/api/tasks/{case.task_id}/task-artifacts").json()
        row = next(r for r in listing["artifacts"] if r["id"] == record["id"])
        assert row["available"] is False and row["download_url"] is None
    read = case.observed["output"]["evidence_url"]
    assert case.rt.maker.get(read).status_code == 200
    assert case.rt.other.get(read).status_code == 403


@pytest.mark.parametrize("kind", ["event", "batch", "reconciliation"])
def test_legacy_sensitive_output_without_invocation_receipt_cannot_skip_read_guard(
    tmp_path, monkeypatch, kind
):
    case = _case(tmp_path, monkeypatch, kind, unbound=True, completion_loss=True)
    assert _records(case) == []
    assert case.run["output_ref"] is not None
    fresh, maker = _fresh(case, monkeypatch)
    other = TestClient(fresh)
    other.cookies.update(case.rt.other.cookies)
    for client in (other, TestClient(fresh), maker):
        for endpoint in ("reconcile", "resume-completion"):
            response = client.post(
                f"/api/plans/{case.plan['id']}/{endpoint}",
                json={"target_id": case.target["id"]},
            )
            assert response.status_code == 403, response.text
    assert (
        case.rt.app.state.plan_repo.load_plan(case.plan["id"]).status.value == "failed"
    )


def test_missing_receipt_file_cannot_skip_cached_resolution_read_guard(
    tmp_path, monkeypatch
):
    case = _case(tmp_path, monkeypatch, cancelled=True)
    response = case.rt.maker.post(
        f"/api/plans/{case.plan['id']}/reconcile", json={"target_id": case.target["id"]}
    )
    assert response.status_code == 200, response.text
    assert response.json()["reconciliation_result"]["outcome"] == "applied"
    assert response.json()["plan"]["status"] == "cancelled"
    Path(_records(case)[0]["path"]).unlink()
    for client in (case.rt.other, case.rt.maker):
        for endpoint in ("reconcile", "resume-completion"):
            response = client.post(
                f"/api/plans/{case.plan['id']}/{endpoint}",
                json={"target_id": case.target["id"]},
            )
            assert response.status_code == 403, response.text
    assert (
        case.rt.app.state.plan_repo.load_plan(case.plan["id"]).status.value
        == "cancelled"
    )


def test_ordinary_legacy_batch_without_source_scope_keeps_completion(
    ordinary_batch, monkeypatch  # noqa: F811
):
    app, material, contract, _, _ = ordinary_batch
    client = TestClient(app)
    ctx = SimpleNamespace(
        rt=SimpleNamespace(maker=client), material=material, contract=contract
    )
    import marvis.plugins.runner as module

    original = module._run_worker

    def legacy(python, job, **kwargs):
        return original(
            python, {k: v for k, v in job.items() if k != "invocation_id"}, **kwargs
        )

    def interrupted(*args):
        raise RuntimeError("synthetic interrupted ordinary completion")

    monkeypatch.setattr(module, "_run_worker", legacy)
    monkeypatch.setattr(
        app.state.plan_executor._reviewer, "deterministic_check", interrupted
    )
    failed = batches.run(ctx, batches.proposal(ctx))
    assert failed["status"] == "failed", failed
    target = next(
        t for t in failed["reconciliation"]["targets"] if t["kind"] == "completion"
    )
    fresh = TestClient(create_app(app.state.settings))
    response = fresh.post(
        f"/api/plans/{failed['id']}/reconcile", json={"target_id": target["id"]}
    )
    assert response.status_code == 200, response.text
    assert response.json()["plan"]["status"] == "done", response.text
