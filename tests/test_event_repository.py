from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from marvis.db_schema import connect, init_db
from marvis.data.errors import DatasetContentDriftError
from marvis.domain import TaskCreate
from marvis.governance.repository import GovernanceRepository
from marvis.production_governance.repository import ProductionGovernanceRepository
from marvis.repositories.task_artifacts import TaskArtifactRepository
from marvis.repositories.tasks import TaskRepository
from marvis.risk_context.event_contracts import (
    EventError,
    EventImport,
    EventSourceGrant,
)
from marvis.risk_context.event_repository import EventRepository, ensure_event_schema
from marvis.risk_context.event_replay import evaluate_event_snapshot
from marvis.settings import build_settings

from test_event_features import T, ago, coverage, event, query, source, values


def test_authorize_read_checks_live_grant_without_loading_snapshot(runtime, monkeypatch):
    rt = runtime
    monkeypatch.setattr(rt.repo, "_decision_body", lambda *a: pytest.fail("loaded snapshot"))
    assert rt.repo.authorize_read(
        rt.task.id, rt.source.source_id,
        grant_id=rt.grant.grant_id, actor_id=rt.actors["maker"],
    ) == rt.source
    with pytest.raises(EventError, match="scope_forbidden"):
        rt.repo.authorize_read(
            rt.task.id, rt.source.source_id,
            grant_id=rt.grant.grant_id, actor_id=rt.actors["other"],
        )
    rt.repo.revoke_grant(rt.task.id, rt.grant.grant_id, rt.actors["admin"])
    with pytest.raises(EventError, match="not_active"):
        rt.repo.authorize_read(
            rt.task.id, rt.source.source_id,
            grant_id=rt.grant.grant_id, actor_id=rt.actors["maker"],
        )


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    settings.plugin_admin_token_path.write_text("synthetic-local-event-secret-" * 4)
    tasks = TaskRepository(settings.db_path)
    task = tasks.create_task(
        TaskCreate(
            model_name="event testing",
            model_version="test",
            validator="test",
            source_dir=str(tmp_path),
        )
    )
    other_task = tasks.create_task(
        TaskCreate(
            model_name="other task",
            model_version="test",
            validator="test",
            source_dir=str(tmp_path),
        )
    )
    governance = GovernanceRepository(settings.db_path)
    production = ProductionGovernanceRepository(settings.db_path)
    actors = {}
    for name, role in (
        ("admin", "admin"),
        ("maker", "maker"),
        ("other", "maker"),
        ("checker", "checker"),
    ):
        actor = governance.create_local_principal(display_name=name)
        production.claim_principal(
            local_principal_id=actor.id, display_name=name, role=role
        )
        actors[name] = actor.id
    basis_path = settings.tasks_dir / task.id / "authorization.json"
    basis_path.parent.mkdir(parents=True)
    basis_path.write_text(json.dumps({"purpose": "deidentified synthetic event test"}))
    basis = TaskArtifactRepository(settings.db_path).register(
        task_id=task.id,
        kind="event_authorization_basis",
        path=str(basis_path),
        content_hash=hashlib.sha256(basis_path.read_bytes()).hexdigest(),
        origin_tool="test_fixture",
        provenance={"declaration": "synthetic_reference_test"},
    )
    repo = EventRepository(settings)
    src = source(task_id=task.id)
    repo.register_source(src, actors["admin"])
    grant = EventSourceGrant(
        grant_id="event-maker-grant",
        task_id=task.id,
        source_id=src.source_id,
        source_contract_hash=src.contract_hash,
        grantee_id=actors["maker"],
        permissions=["read", "write"],
        purpose="review event features",
        basis_artifact_id=basis["id"],
        starts_at="2020-01-01T00:00:00Z",
        expires_at="2100-01-01T00:00:00Z",
    )
    repo.create_grant(grant, actors["admin"])
    clock = {"at": ago(1)}
    monkeypatch.setattr(
        "marvis.risk_context.event_repository._now", lambda: clock["at"]
    )
    return SimpleNamespace(
        repo=repo,
        settings=settings,
        task=task,
        other_task=other_task,
        actors=actors,
        source=src,
        grant=grant,
        basis_path=basis_path,
        clock=clock,
        serial=itertools.count(),
        path=tmp_path,
    )


def dataset(rt, rows, *, column="claim", owner=None):
    path = rt.path / f"event-import-{next(rt.serial)}.parquet"
    pd.DataFrame({column: rows}).to_parquet(path, index=False)
    return rt.repo.registry.register_existing(
        path, task_id=owner or rt.task.id, role="event_history"
    )


def request(rt, rows, *, kind="events", column="claim", owner=None):
    data = dataset(
        rt, [json.dumps(row.model_dump()) for row in rows], column=column, owner=owner
    )
    return EventImport(
        source_id=rt.source.source_id,
        source_contract_hash=rt.source.contract_hash,
        dataset_id=data.id,
        expected_content_hash=data.content_hash,
        kind=kind,
        json_column=column,
    )


def ingest(rt, rows, **options):
    req = request(rt, rows, **options)
    return rt.repo.ingest_dataset(
        rt.task.id, req, grant_id=rt.grant.grant_id, actor_id=rt.actors["maker"]
    )


def evaluate(rt, decision_id="decision-1", contract=None):
    rt.clock["at"] = max(rt.clock["at"], T)
    return rt.repo.evaluate(
        rt.task.id,
        decision_id,
        contract or query(rt.source),
        grant_id=rt.grant.grant_id,
        actor_id=rt.actors["maker"],
    )


def claims_count(rt):
    with connect(rt.settings.db_path) as conn:
        return conn.execute("SELECT count(*) FROM event_claims").fetchone()[0]


def test_actual_registry_sqlite_native_snapshot_and_restart_replay(runtime):
    rt = runtime
    ingest(rt, [event(), event("evt-2", values={"amount_minor": 2**53 + 1})])
    ingest(rt, [coverage()], kind="coverage")
    original = evaluate(rt)
    assert values(original["result"])["amount"] == 2**53 + 101
    assert (
        original["snapshot"]["claims"][0]["origin"]
        == "authenticated_registered_dataset"
    )
    assert all(
        len(row["dataset_content_hash"]) == 64 for row in original["snapshot"]["claims"]
    )
    restarted = EventRepository(rt.settings)
    replay = restarted.replay(
        rt.task.id,
        "decision-1",
        grant_id=rt.grant.grant_id,
        actor_id=rt.actors["maker"],
    )
    assert replay == original
    assert (
        restarted.evaluate(
            rt.task.id,
            "decision-1",
            query(rt.source),
            grant_id=rt.grant.grant_id,
            actor_id=rt.actors["maker"],
        )
        == original
    )
    from marvis.risk_context.event_contracts import EventSnapshot

    assert (
        evaluate_event_snapshot(EventSnapshot.model_validate(replay["snapshot"]))
        == original["result"]
    )


def test_exact_duplicate_import_is_idempotent_across_threads_and_restart(runtime):
    rt = runtime
    req = request(rt, [event(), event("evt-2")])

    def run():
        return EventRepository(rt.settings).ingest_dataset(
            rt.task.id, req, grant_id=rt.grant.grant_id, actor_id=rt.actors["maker"]
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: run(), range(2)))
    assert results[0] == results[1] == run()
    assert claims_count(rt) == 2
    repeated = ingest(rt, [event(), event("evt-2")], column="different_json_column")
    assert repeated["duplicate_count"] == 2 and repeated["inserted_count"] == 0
    assert claims_count(rt) == 2


def test_conflicting_version_rolls_back_whole_import_without_overwrite(runtime):
    rt = runtime
    ingest(rt, [event()])
    req = request(rt, [event("would-be-new"), event(values={"amount_minor": 900})])
    with pytest.raises(EventError, match="version_content_conflict"):
        rt.repo.ingest_dataset(
            rt.task.id, req, grant_id=rt.grant.grant_id, actor_id=rt.actors["maker"]
        )
    assert claims_count(rt) == 1


def test_late_correction_and_out_of_order_predecessor_never_change_original(runtime):
    rt = runtime
    second = event(
        version=2, supersedes_version=1, available_at=T, values={"amount_minor": 800}
    )
    ingest(rt, [second])
    ingest(rt, [coverage()], kind="coverage")
    incomplete = evaluate(rt)
    assert values(incomplete["result"])["amount"] is None
    assert (
        "version_lineage_incomplete"
        in incomplete["result"]["features"]["amount"]["missing_reasons"]
    )
    rt.clock["at"] = ago(-10)
    ingest(rt, [event()])
    assert evaluate(rt) == incomplete
    assert values(evaluate(rt, "same-time-new-id")["result"])["amount"] is None
    retrospective = query(
        rt.source, availability_mode="retrospective_declared", knowledge_cutoff=ago(-10)
    )
    corrected = evaluate(rt, "later-retrospective", retrospective)
    assert values(corrected["result"])["amount"] == 800
    assert corrected["result"]["historical_platform_visibility"] == "not_asserted"
    assert (
        rt.repo.replay(
            rt.task.id,
            "decision-1",
            grant_id=rt.grant.grant_id,
            actor_id=rt.actors["maker"],
        )
        == incomplete
    )
    with pytest.raises(EventError, match="contract_conflict"):
        evaluate(rt, contract=retrospective)


def test_backdated_late_version_cannot_enter_original_platform_snapshot(runtime):
    rt = runtime
    ingest(rt, [event()])
    ingest(rt, [coverage()], kind="coverage")
    original = evaluate(rt)
    rt.clock["at"] = ago(-10)
    ingest(
        rt,
        [
            event(
                version=2,
                supersedes_version=1,
                available_at=ago(5),
                values={"amount_minor": 800},
            )
        ],
    )
    assert evaluate(rt) == original
    assert values(evaluate(rt, "new-key-same-cutoff")["result"])["amount"] == 100


def test_available_time_regression_is_a_conflict(runtime):
    rt = runtime
    ingest(rt, [event()])
    with pytest.raises(EventError, match="availability_conflict"):
        ingest(
            rt,
            [
                event(
                    version=2,
                    supersedes_version=1,
                    event_at=ago(30),
                    available_at=ago(20),
                )
            ],
        )
    assert claims_count(rt) == 1


def test_missing_availability_is_persisted_as_unknown_not_approval(runtime):
    rt = runtime
    ingest(rt, [event(available_at=None)])
    ingest(rt, [coverage()], kind="coverage")
    result = evaluate(rt)["result"]
    assert result["status"] == "unknown" and result["next_action"] == "required_review"
    assert set(values(result).values()) == {None}
    assert result["automated_clearance"] is False


def test_empty_source_needs_explicit_coverage_to_return_zero(runtime):
    rt = runtime
    empty = evaluate(rt)
    assert set(values(empty["result"]).values()) == {None}
    ingest(rt, [coverage()], kind="coverage")
    assert set(values(evaluate(rt, "complete-empty")["result"]).values()) == {0}
    assert evaluate(rt) == empty


def test_task_source_and_grantee_scope_are_enforced(runtime):
    rt = runtime
    req = request(rt, [event()])
    with pytest.raises(EventError, match="scope_forbidden"):
        rt.repo.ingest_dataset(
            rt.task.id, req, grant_id=rt.grant.grant_id, actor_id=rt.actors["other"]
        )
    with pytest.raises(EventError, match="grant_not_found"):
        rt.repo.ingest_dataset(
            rt.other_task.id,
            req,
            grant_id=rt.grant.grant_id,
            actor_id=rt.actors["maker"],
        )
    with pytest.raises(EventError, match="source_version_conflict"):
        rt.repo.register_source(source(task_id=rt.other_task.id), rt.actors["admin"])
    foreign = request(rt, [event()], owner=rt.other_task.id)
    with pytest.raises(DatasetContentDriftError, match="expected task"):
        rt.repo.ingest_dataset(
            rt.task.id, foreign, grant_id=rt.grant.grant_id, actor_id=rt.actors["maker"]
        )
    assert claims_count(rt) == 0


def test_shared_read_grant_and_revocation_leave_old_evidence_unchanged(runtime):
    rt = runtime
    ingest(rt, [event()])
    ingest(rt, [coverage()], kind="coverage")
    original = evaluate(rt)
    read_grant = rt.grant.model_copy(
        update={
            "grant_id": "checker-read",
            "grantee_id": rt.actors["checker"],
            "permissions": ["read"],
        }
    )
    rt.repo.create_grant(read_grant, rt.actors["admin"])
    assert (
        rt.repo.replay(
            rt.task.id,
            "decision-1",
            grant_id=read_grant.grant_id,
            actor_id=rt.actors["checker"],
        )
        == original
    )
    with pytest.raises(ValueError, match="role_forbidden"):
        rt.repo.ingest_dataset(
            rt.task.id,
            request(rt, [event("new")]),
            grant_id=read_grant.grant_id,
            actor_id=rt.actors["checker"],
        )
    with pytest.raises(ValueError, match="role_forbidden"):
        rt.repo.evaluate(
            rt.task.id,
            "checker-cannot-publish",
            query(rt.source),
            grant_id=read_grant.grant_id,
            actor_id=rt.actors["checker"],
        )
    rt.repo.revoke_grant(rt.task.id, rt.grant.grant_id, rt.actors["admin"])
    with pytest.raises(EventError, match="not_active"):
        evaluate(rt)
    assert (
        rt.repo.replay(
            rt.task.id,
            "decision-1",
            grant_id=read_grant.grant_id,
            actor_id=rt.actors["checker"],
        )
        == original
    )


def test_read_permission_does_not_allow_a_maker_to_publish_new_snapshot(runtime):
    rt = runtime
    read_grant = rt.grant.model_copy(
        update={"grant_id": "maker-read-only", "permissions": ["read"]}
    )
    rt.repo.create_grant(read_grant, rt.actors["admin"])
    with pytest.raises(EventError, match="scope_forbidden"):
        rt.repo.evaluate(
            rt.task.id,
            "new",
            query(rt.source),
            grant_id=read_grant.grant_id,
            actor_id=rt.actors["maker"],
        )
    rt.repo.revoke_grant(rt.task.id, read_grant.grant_id, rt.actors["admin"])
    with pytest.raises(EventError, match="version_conflict"):
        rt.repo.create_grant(read_grant, rt.actors["admin"])


def test_grant_basis_requires_same_task_and_unchanged_registered_bytes(runtime):
    rt = runtime
    with pytest.raises(EventError, match="basis_not_found"):
        rt.repo.create_grant(
            rt.grant.model_copy(
                update={"grant_id": "bad", "basis_artifact_id": "missing"}
            ),
            rt.actors["admin"],
        )
    rt.basis_path.write_text("changed authorization")
    with pytest.raises(EventError, match="basis_integrity_failed"):
        evaluate(rt)


@pytest.mark.parametrize("change", ["revoked", "expired", "production_inactive"])
def test_active_role_is_rechecked_on_every_operation(runtime, change):
    rt = runtime
    with connect(rt.settings.db_path) as conn:
        if change == "revoked":
            conn.execute(
                "UPDATE local_principals SET status='revoked' WHERE id=?",
                (rt.actors["maker"],),
            )
        elif change == "expired":
            conn.execute(
                "UPDATE local_principals SET expires_at='2020-01-01T00:00:00+00:00' WHERE id=?",
                (rt.actors["maker"],),
            )
        else:
            conn.execute(
                "UPDATE production_principals SET status='revoked' WHERE local_principal_id=?",
                (rt.actors["maker"],),
            )
    with pytest.raises(ValueError, match="forbidden|expired|inactive"):
        evaluate(rt)


def test_registered_dataset_tampering_fails_before_append(runtime):
    rt = runtime
    req = request(rt, [event()])
    binding = rt.repo.registry.authenticate_dataset_binding(
        req.dataset_id,
        expected_task_id=rt.task.id,
        expected_content_hash=req.expected_content_hash,
    )
    original = binding.path.read_bytes()
    original_mode = binding.path.stat().st_mode
    try:
        binding.path.chmod(0o600)
        binding.path.write_bytes(b"tampered")
        with pytest.raises(DatasetContentDriftError, match="integrity"):
            rt.repo.ingest_dataset(
                rt.task.id, req, grant_id=rt.grant.grant_id, actor_id=rt.actors["maker"]
            )
    finally:
        binding.path.write_bytes(original)
        binding.path.chmod(original_mode)
    assert claims_count(rt) == 0


def test_dataset_binding_is_rechecked_inside_write_transaction(runtime, monkeypatch):
    rt = runtime
    req = request(rt, [event()])
    read = rt.repo.registry.read_authenticated_binding_snapshot

    def change_after_read(binding, **kwargs):
        frame = read(binding, **kwargs)
        with connect(rt.settings.db_path) as conn:
            conn.execute(
                "UPDATE datasets SET content_hash=? WHERE id=?",
                ("f" * 64, req.dataset_id),
            )
        return frame

    monkeypatch.setattr(
        rt.repo.registry, "read_authenticated_binding_snapshot", change_after_read
    )
    with pytest.raises(DatasetContentDriftError, match="binding changed before commit"):
        rt.repo.ingest_dataset(
            rt.task.id, req, grant_id=rt.grant.grant_id, actor_id=rt.actors["maker"]
        )
    assert claims_count(rt) == 0


def test_explicit_budgets_fail_without_partial_append_or_truncated_decision(
    runtime, monkeypatch
):
    rt = runtime
    req = request(rt, [event(), event("two")])
    with monkeypatch.context() as patch:
        patch.setattr("marvis.risk_context.event_repository.MAX_EVENTS", 1)
        with pytest.raises(EventError, match="row_budget"):
            rt.repo.ingest_dataset(
                rt.task.id, req, grant_id=rt.grant.grant_id, actor_id=rt.actors["maker"]
            )
    assert claims_count(rt) == 0
    rt.repo.ingest_dataset(
        rt.task.id, req, grant_id=rt.grant.grant_id, actor_id=rt.actors["maker"]
    )
    with monkeypatch.context() as patch:
        patch.setattr("marvis.risk_context.event_repository.MAX_SNAPSHOT_BYTES", 1)
        with pytest.raises(EventError, match="snapshot_budget"):
            evaluate(rt)
    with connect(rt.settings.db_path) as conn:
        assert conn.execute("SELECT count(*) FROM event_decisions").fetchone()[0] == 0


@pytest.mark.parametrize(
    "raw",
    [
        json.dumps({**event().model_dump(), "event_id": 2**53 + 1}),
        json.dumps({**event().model_dump(), "event_id": float(2**53 + 1)}),
        '{"event_id":"a","event_id":"b"}',
        "not-json",
    ],
)
def test_invalid_or_lossy_identity_json_cannot_be_ingested(runtime, raw):
    rt = runtime
    data = dataset(rt, [raw])
    req = EventImport(
        source_id=rt.source.source_id,
        source_contract_hash=rt.source.contract_hash,
        dataset_id=data.id,
        expected_content_hash=data.content_hash,
        kind="events",
        json_column="claim",
    )
    with pytest.raises(ValueError):
        rt.repo.ingest_dataset(
            rt.task.id, req, grant_id=rt.grant.grant_id, actor_id=rt.actors["maker"]
        )
    assert claims_count(rt) == 0


def test_event_schema_does_not_change_global_version_and_evidence_is_immutable(runtime):
    rt = runtime
    with connect(rt.settings.db_path) as conn:
        before = conn.execute("PRAGMA user_version").fetchone()[0]
    ensure_event_schema(rt.settings.db_path)
    ingest(rt, [event()])
    original = evaluate(rt)
    with connect(rt.settings.db_path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == before
        with pytest.raises(Exception, match="immutable"):
            conn.execute("UPDATE event_claims SET payload='{}'")
    with connect(rt.settings.db_path) as conn:
        conn.execute("DROP TRIGGER event_decisions_immutable")
        body = dict(original)
        body["result"] = {**body["result"], "automated_clearance": True}
        conn.execute("UPDATE event_decisions SET payload=?", (json.dumps(body),))
    with pytest.raises(EventError, match="integrity_failed"):
        rt.repo.replay(
            rt.task.id,
            "decision-1",
            grant_id=rt.grant.grant_id,
            actor_id=rt.actors["maker"],
        )


@pytest.mark.parametrize(
    "table",
    [
        "event_sources",
        "event_grants",
        "event_claims",
        "event_imports",
        "event_decisions",
    ],
)
def test_live_event_evidence_cannot_be_deleted_but_task_cascade_is_allowed(
    runtime, table
):
    import sqlite3

    rt = runtime
    ingest(rt, [event()])
    ingest(rt, [coverage()], kind="coverage")
    evaluate(rt)
    with connect(rt.settings.db_path) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="event evidence is immutable"):
            conn.execute(f"DELETE FROM {table}")
    with connect(rt.settings.db_path) as conn:
        conn.execute("DELETE FROM tasks WHERE id=?", (rt.task.id,))
        assert conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0
