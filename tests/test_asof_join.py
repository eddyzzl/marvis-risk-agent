from dataclasses import replace
import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

import marvis.data.asof_join as asof_module
from marvis.data.asof_join import AsOfJoinEngine
from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.data.time_contracts import DatasetTimeContract
from marvis.db import DatasetRepository, TaskRepository, init_db
from marvis.domain import TaskCreate
from marvis.files import sha256_file
from marvis.repositories.task_artifacts import TaskArtifactRepository
from tests.test_data_time_contracts import contracts, frames, spec


@pytest.fixture
def env(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    task = TaskRepository(db_path).create_task(TaskCreate("pit", "v1", "test", "/synthetic"))
    root = tmp_path / "datasets"
    repo = DatasetRepository(db_path)
    registry = DatasetRegistry(repo, DataBackend(root), root)
    artifacts = TaskArtifactRepository(db_path)
    engine = AsOfJoinEngine(registry, artifacts, workspace_root=tmp_path)
    datasets = []
    for name, frame in zip(("decisions", "features"), frames(), strict=True):
        path = root / task.id / f"{name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        with repo.transaction() as conn:
            datasets.append(registry.register_existing_on_connection(
                conn, path, task_id=task.id, role=name, target_col_override=None,
            ))
    dc, fc = [DatasetTimeContract.model_validate({
        **contract.model_dump(), "dataset_id": dataset.id, "content_hash": dataset.content_hash,
    }) for dataset, contract in zip(datasets, contracts(), strict=True)]
    return engine, task.id, dc, fc, repo


def run(env, **kwargs):
    engine, task_id, dc, fc, _ = env
    return engine.execute(task_id=task_id, decision_contract=dc, feature_contract=fc, spec=spec(**kwargs))


def output_files(engine):
    return [p for p in engine.root.rglob("*") if p.is_file() and ".pit" in p.parts]


def test_real_asof_registration_binds_two_authenticated_parents_and_is_replayable(env):
    engine, task_id, dc, fc, _ = env
    original = [(engine.registry.get(c.dataset_id).content_hash, engine.registry.get(c.dataset_id).created_at)
                for c in (dc, fc)]
    first = run(env)
    second = run(env)
    assert first.dataset.id != second.dataset.id
    assert first.dataset.content_hash == second.dataset.content_hash
    assert first.membership_sha256 == second.membership_sha256
    assert first.status.assurance == "verified"
    assert engine.dataset_time_status(first.dataset.id) == first.status
    frame = engine.registry.read_authenticated_parquet_snapshot(first.dataset.id)
    assert frame.asof__amount.tolist() == [10, 20]
    proof = json.loads(first.evidence_path.read_text())
    assert proof["memberships"] == [[0, 0], [1, 1]]
    assert proof["output_sha256"] == first.dataset.content_hash
    records = engine.artifacts.list_for_task(task_id)
    assert len(records) == 2
    assert [p["dataset_id"] for p in records[0]["provenance"]["parents"]] == [dc.dataset_id, fc.dataset_id]
    for c, before in zip((dc, fc), original, strict=True):
        ds = engine.registry.get(c.dataset_id)
        assert (ds.content_hash, ds.created_at) == before
        assert ds.source_path.startswith("_cas/")


def test_old_and_ordinary_join_data_remain_unknown_without_rewriting_bytes_or_hash(env):
    engine, _, dc, _, repo = env
    original = engine.registry.get(dc.dataset_id)
    path = engine.root / original.source_path
    before = path.read_bytes()
    assert engine.dataset_time_status(original.id).assurance == "unknown"
    ordinary = replace(original, id="ordinary-join", role="joined")
    repo.create_dataset(ordinary)
    assert engine.dataset_time_status(ordinary.id).reasons == ("no_point_in_time_evidence",)
    assert engine.registry.get(original.id) == original
    assert path.read_bytes() == before
    assert engine.artifacts.list_for_task(original.task_id) == []


def test_unknown_historical_availability_cannot_be_verified_or_filled_from_created_at(env):
    engine, task_id, dc, fc, _ = env
    fc = DatasetTimeContract.model_validate({**fc.model_dump(), "available_at": None})
    with pytest.raises(ValueError, match="recorded historical"):
        engine.execute(task_id=task_id, decision_contract=dc, feature_contract=fc, spec=spec())
    result = engine.execute(task_id=task_id, decision_contract=dc, feature_contract=fc, spec=spec(mode="exploration"))
    assert result.status.assurance == "unknown"
    assert engine.dataset_time_status(result.dataset.id).assurance == "unknown"
    assert json.loads(result.evidence_path.read_text())["feature_contract"]["available_at"] is None


def test_registered_target_may_not_enter_feature_selection(env):
    engine, _, _, fc, repo = env
    with repo.transaction() as conn:
        conn.execute("UPDATE datasets SET target_col='amount', has_target=1 WHERE id=?", (fc.dataset_id,))
    with pytest.raises(ValueError, match="registered target"):
        run(env)
    assert not output_files(engine)


@pytest.mark.parametrize("phase", ["before", "after_compute", "during_commit"])
def test_source_identity_drift_at_each_boundary_fails_without_output_or_artifact(env, monkeypatch, phase):
    engine, task_id, _, fc, repo = env
    def drift():
        with repo.transaction() as conn:
            conn.execute("UPDATE datasets SET content_hash=? WHERE id=?", ("f" * 64, fc.dataset_id))
    if phase == "before":
        drift()
    elif phase == "after_compute":
        original = asof_module.select_asof_rows
        def selection(*args):
            result = original(*args)
            drift()
            return result
        monkeypatch.setattr(asof_module, "select_asof_rows", selection)
    else:
        original = engine.registry.register_existing_on_connection
        def register(conn, *args, **kwargs):
            result = original(conn, *args, **kwargs)
            conn.execute("UPDATE datasets SET content_hash=? WHERE id=?", ("f" * 64, fc.dataset_id))
            return result
        monkeypatch.setattr(engine.registry, "register_existing_on_connection", register)
    with pytest.raises(Exception, match="content hash|binding changed"):
        run(env)
    assert len(engine.registry.list_for_task(task_id)) == 2
    assert engine.artifacts.list_for_task(task_id) == []
    assert not output_files(engine)


def test_artifact_registration_failure_rolls_back_matrix_and_dataset_row(env, monkeypatch):
    engine, task_id, _, _, _ = env
    def fail(*args, **kwargs):
        raise ValueError("injected artifact conflict")
    monkeypatch.setattr(engine.artifacts, "register_on_connection", fail)
    with pytest.raises(ValueError, match="artifact conflict"):
        run(env)
    assert len(engine.registry.list_for_task(task_id)) == 2
    assert not output_files(engine)


@pytest.mark.parametrize("target", ["evidence", "output", "source", "source_row", "target"])
def test_consumption_reverifies_all_materialized_evidence(env, target):
    engine, _, _, fc, repo = env
    result = run(env)
    if target == "evidence":
        result.evidence_path.write_text("{}")
    elif target == "output":
        path = engine.root / result.dataset.source_path
        path.write_bytes(b"changed")
    elif target == "source":
        source = engine.registry.get(fc.dataset_id)
        path = engine.root / source.source_path
        path.chmod(0o644)
        path.write_bytes(b"changed")
    elif target == "source_row":
        with repo.transaction() as conn:
            conn.execute("UPDATE datasets SET content_hash=? WHERE id=?", ("f" * 64, fc.dataset_id))
    else:
        with repo.transaction() as conn:
            conn.execute("UPDATE datasets SET target_col='event', has_target=1 WHERE id=?", (fc.dataset_id,))
    with pytest.raises(Exception, match="changed|failed integrity|provenance"):
        engine.dataset_time_status(result.dataset.id)


def test_snapshot_hash_mismatch_is_not_repaired_by_contract(env):
    engine, task_id, dc, fc, _ = env
    bad = DatasetTimeContract.model_validate({**fc.model_dump(), "content_hash": "f" * 64})
    with pytest.raises(Exception, match="content hash"):
        engine.execute(task_id=task_id, decision_contract=dc, feature_contract=bad, spec=spec())
    assert not output_files(engine)


def test_materialization_rechecks_file_after_profiling_before_commit(env, monkeypatch):
    engine, task_id, _, _, _ = env
    original = engine.registry.register_existing_on_connection
    def register(conn, path, **kwargs):
        result = original(conn, path, **kwargs)
        Path(path).write_bytes(b"changed after profile")
        return result
    monkeypatch.setattr(engine.registry, "register_existing_on_connection", register)
    with pytest.raises(ValueError, match="changed before commit"):
        run(env)
    assert len(engine.registry.list_for_task(task_id)) == 2
    assert engine.artifacts.list_for_task(task_id) == []
    assert not output_files(engine)


def test_contract_hash_mismatch_after_real_source_mutation_fails_closed(env):
    engine, _, _, fc, _ = env
    path = engine.root / engine.registry.get(fc.dataset_id).source_path
    frame = pd.read_parquet(path)
    frame.loc[0, "amount"] = 777
    frame.to_parquet(path, index=False)
    assert sha256_file(path) != fc.content_hash
    with pytest.raises(Exception, match="failed integrity"):
        run(env)


def test_provenance_is_immutable_at_database_boundary(env):
    engine, _, _, _, repo = env
    result = run(env)
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        with repo.transaction() as conn:
            conn.execute("UPDATE task_artifacts SET provenance_json='{}' WHERE id=?", (result.status.artifact_id,))
    assert engine.dataset_time_status(result.dataset.id).assurance == "verified"


def test_cross_database_artifact_registry_rejected(env, tmp_path):
    engine = env[0]
    with pytest.raises(ValueError, match="share one database"):
        AsOfJoinEngine(engine.registry, TaskArtifactRepository(tmp_path / "other.db"), workspace_root=tmp_path)


def test_output_cannot_escape_through_existing_task_directory_symlink(env, tmp_path):
    engine, task_id, _, _, _ = env
    outside = tmp_path / "outside"
    outside.mkdir()
    # Existing source files are pinned first, then the abandoned upload directory
    # can be replaced to exercise the output path without touching source bytes.
    for contract in (env[2], env[3]):
        engine.registry.authenticate_dataset_binding(contract.dataset_id, expected_task_id=task_id,
                                                     expected_content_hash=contract.content_hash)
    original = engine.root / task_id
    original.rename(engine.root / "old-source-uploads")
    original.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        run(env)
    assert list(outside.iterdir()) == []


def test_exact_matrix_integrity_does_not_allow_float_tolerance_or_expose_values():
    with pytest.raises(ValueError, match="materialized matrix differs") as error:
        asof_module._verify_matrix(pd.DataFrame({"secret": [12345.0]}), pd.DataFrame({"secret": [12345.00001]}))
    assert "12345" not in str(error.value)
