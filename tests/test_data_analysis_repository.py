from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import sqlite3
from threading import Barrier

import pytest

from marvis.data.workspace import (
    DataSemanticMapping,
    data_semantic_mapping_hash,
    data_semantic_mapping_to_dict,
)
from marvis.db_schema import connect, init_db
from marvis.repositories.data_analysis import (
    DATA_ANALYSIS_ARTIFACT_ORIGIN_TOOL,
    DATA_ANALYSIS_SCHEMA_VERSION,
    DataAnalysisConflictError,
    DataAnalysisDataError,
    DataAnalysisIdentity,
    DataAnalysisNotFoundError,
    DataAnalysisRepository,
    DataAnalysisStaleIdentityError,
    DataAnalysisTransitionError,
)
from marvis.repositories.task_artifacts import TaskArtifactRepository


_CREATED_AT = "2026-07-19T00:00:00+00:00"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _mapping() -> DataSemanticMapping:
    return DataSemanticMapping(
        target_col="bad",
        field_roles={"bad": "target", "score": "score"},
        business_names={"bad": "是否逾期", "score": "模型分"},
    )


def _seed_domain(db_path) -> None:
    init_db(db_path)
    mapping = _mapping()
    mapping_json = _canonical_json(data_semantic_mapping_to_dict(mapping))
    with connect(db_path) as conn:
        for task_id in ("task-1", "task-2"):
            conn.execute(
                """
                INSERT INTO tasks(
                    id, model_name, model_version, validator, source_dir,
                    status, status_message, created_at, updated_at
                ) VALUES (?, 'analysis task', 'v1', 'tester', '/tmp/source',
                          'draft', '', ?, ?)
                """,
                (task_id, _CREATED_AT, _CREATED_AT),
            )
        conn.executemany(
            """
            INSERT INTO datasets(
                id, task_id, role, source_path, format, row_count, columns_json,
                has_target, target_col, created_at, content_hash
            ) VALUES (?, ?, 'analysis', ?, 'csv', 12, ?, 1, 'bad', ?, ?)
            """,
            [
                (
                    "dataset-1",
                    "task-1",
                    "/tmp/dataset-1.csv",
                    '["score","bad"]',
                    _CREATED_AT,
                    _sha("dataset-1"),
                ),
                (
                    "dataset-2",
                    "task-2",
                    "/tmp/dataset-2.csv",
                    '["score","bad"]',
                    _CREATED_AT,
                    _sha("dataset-2"),
                ),
            ],
        )
        conn.executemany(
            """
            INSERT INTO data_workspaces(
                task_id, schema_version, revision, active_dataset_id,
                active_dataset_content_hash, analysis_generation, page,
                selected_field, semantic_mapping_json, updated_at
            ) VALUES (?, 'data-workspace.v1', 3, ?, ?, 1, 'statistics',
                      'score', ?, ?)
            """,
            [
                (
                    "task-1",
                    "dataset-1",
                    _sha("dataset-1"),
                    mapping_json,
                    _CREATED_AT,
                ),
                (
                    "task-2",
                    "dataset-2",
                    _sha("dataset-2"),
                    mapping_json,
                    _CREATED_AT,
                ),
            ],
        )


def _identity(
    *,
    task_id: str = "task-1",
    dataset_id: str = "dataset-1",
    dataset_content_hash: str | None = None,
    workspace_revision: int = 3,
    analysis_generation: int = 1,
    semantic_mapping_hash: str | None = None,
    config: dict | None = None,
    producer_version: str = DATA_ANALYSIS_SCHEMA_VERSION,
) -> DataAnalysisIdentity:
    return DataAnalysisIdentity(
        task_id=task_id,
        dataset_id=dataset_id,
        dataset_content_hash=(
            dataset_content_hash
            if dataset_content_hash is not None
            else _sha(dataset_id)
        ),
        workspace_revision=workspace_revision,
        analysis_generation=analysis_generation,
        semantic_mapping_hash=(
            semantic_mapping_hash
            if semantic_mapping_hash is not None
            else data_semantic_mapping_hash(_mapping())
        ),
        config=config or {"histogram_bins": 20, "columns": ["bad", "score"]},
        producer_version=producer_version,
    )


def _seed_job(
    db_path,
    *,
    job_id: str,
    task_id: str,
    kind: str = "data_analysis",
    status: str = "queued",
) -> None:
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO jobs(id, task_id, kind, status, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (job_id, task_id, kind, status, _CREATED_AT),
        )


def _artifact_provenance(run) -> dict[str, object]:
    return {
        "schema_version": run.schema_version,
        "task_id": run.task_id,
        "dataset_id": run.dataset_id,
        "dataset_content_hash": run.dataset_content_hash,
        "analysis_generation": run.analysis_generation,
        "semantic_mapping_hash": run.semantic_mapping_hash,
        "config_hash": run.config_hash,
        "producer_version": run.producer_version,
        "input_hash": run.input_hash,
    }


def test_identity_uses_canonical_config_and_compute_hash_ignores_page_revision():
    first = _identity(
        config={"columns": ["bad", "score"], "histogram_bins": 20},
    )
    reordered = _identity(
        workspace_revision=4,
        config={"histogram_bins": 20, "columns": ["bad", "score"]},
    )

    assert first.schema_version == DATA_ANALYSIS_SCHEMA_VERSION == "data-analysis.v1"
    assert first.config_json == '{"columns":["bad","score"],"histogram_bins":20}'
    assert first.config_hash == _sha(first.config_json)
    assert reordered.config_json == first.config_json
    assert reordered.input_hash == first.input_hash
    assert reordered.workspace_revision != first.workspace_revision
    assert _identity(config={"histogram_bins": 21}).input_hash != first.input_hash


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"task_id": " task-1"}, "task_id"),
        ({"dataset_content_hash": "A" * 64}, "dataset_content_hash"),
        ({"workspace_revision": True}, "workspace_revision"),
        ({"analysis_generation": -1}, "analysis_generation"),
        ({"semantic_mapping_hash": "bad"}, "semantic_mapping_hash"),
        ({"producer_version": "data-analysis.v1 ",}, "producer_version"),
        ({"config": {"bad": float("nan")}}, "config"),
    ],
)
def test_identity_rejects_noncanonical_input(override, match):
    with pytest.raises(DataAnalysisDataError, match=match):
        _identity(**override)


def test_create_or_get_is_stable_idempotent_and_task_scoped(tmp_path):
    db_path = tmp_path / "app.sqlite"
    _seed_domain(db_path)
    repo = DataAnalysisRepository(db_path)
    identity = _identity(config={"columns": ["bad", "score"], "histogram_bins": 20})

    first = repo.create_or_get(identity)
    replay = repo.create_or_get(_identity())

    assert replay == first
    assert first.identity == identity
    assert first.status == "queued"
    assert first.job_id is None
    assert first.result_artifact_id is None
    assert first.error_kind is None
    assert repo.get_for_task("task-1", first.id) == first
    assert repo.get_for_task("task-2", first.id) is None
    assert repo.get_by_input_hash("task-1", identity.input_hash) == first
    assert repo.current(identity) == first
    assert repo.list_for_task("task-1") == [first]
    assert repo.list_for_task("task-2") == []

    with connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM data_analysis_runs").fetchone()[0]
    assert count == 1


def test_page_only_workspace_save_reuses_prior_computation(tmp_path):
    db_path = tmp_path / "app.sqlite"
    _seed_domain(db_path)
    repo = DataAnalysisRepository(db_path)
    first = repo.create_or_get(_identity())

    with connect(db_path) as conn:
        conn.execute(
            """
            UPDATE data_workspaces
               SET revision = 4, page = 'fields', selected_field = 'bad'
             WHERE task_id = 'task-1'
            """
        )

    replay = repo.create_or_get(_identity(workspace_revision=4))

    assert replay == first
    assert replay.identity.workspace_revision == 3
    assert repo.current(_identity(workspace_revision=4)) == first


def test_create_rejects_stale_workspace_and_dataset_evidence(tmp_path):
    db_path = tmp_path / "app.sqlite"
    _seed_domain(db_path)
    repo = DataAnalysisRepository(db_path)

    with pytest.raises(DataAnalysisStaleIdentityError, match="workspace revision"):
        repo.create_or_get(_identity(workspace_revision=2))
    with pytest.raises(DataAnalysisDataError, match="registered content hash"):
        repo.create_or_get(_identity(dataset_content_hash=_sha("drift")))

    with connect(db_path) as conn:
        conn.execute(
            "UPDATE datasets SET content_hash = ? WHERE id = 'dataset-1'",
            (_sha("registry-drift"),),
        )
    with pytest.raises(DataAnalysisDataError, match="registered content hash"):
        repo.create_or_get(_identity())


def test_create_conceals_cross_task_dataset_and_job_ownership(tmp_path):
    db_path = tmp_path / "app.sqlite"
    _seed_domain(db_path)
    _seed_job(db_path, job_id="job-1", task_id="task-1")
    _seed_job(db_path, job_id="job-2", task_id="task-2")
    repo = DataAnalysisRepository(db_path)

    with pytest.raises(DataAnalysisNotFoundError, match="dataset not found for task"):
        repo.create_or_get(
            _identity(
                task_id="task-2",
                dataset_id="dataset-1",
                dataset_content_hash=_sha("dataset-1"),
            )
        )
    with pytest.raises(DataAnalysisNotFoundError, match="job not found for task"):
        repo.create_or_get(_identity(), job_id="job-2")

    created = repo.create_or_get(_identity(), job_id="job-1")
    assert created.job_id == "job-1"
    assert repo.create_or_get(_identity()) == created
    with connect(db_path) as conn:
        conn.execute("UPDATE jobs SET status = 'failed' WHERE id = 'job-1'")
    _seed_job(db_path, job_id="job-3", task_id="task-1")
    with pytest.raises(DataAnalysisConflictError, match="job_id"):
        repo.create_or_get(_identity(), job_id="job-3")


def test_mark_running_atomically_attaches_a_task_owned_job(tmp_path):
    db_path = tmp_path / "app.sqlite"
    _seed_domain(db_path)
    _seed_job(db_path, job_id="job-1", task_id="task-1")
    repo = DataAnalysisRepository(db_path)
    queued = repo.create_or_get(_identity())
    with connect(db_path) as conn:
        conn.execute("UPDATE jobs SET status = 'running' WHERE id = 'job-1'")

    running = repo.mark_running(
        task_id="task-1",
        run_id=queued.id,
        job_id="job-1",
    )

    assert running.status == "running"
    assert running.job_id == "job-1"
    assert repo.create_or_get(_identity()) == running


def test_attach_job_is_queued_only_idempotent_and_cas_guarded(tmp_path):
    db_path = tmp_path / "app.sqlite"
    _seed_domain(db_path)
    _seed_job(db_path, job_id="job-1", task_id="task-1")
    repo = DataAnalysisRepository(db_path)
    queued = repo.create_or_get(_identity())

    attached = repo.attach_job(
        task_id="task-1",
        run_id=queued.id,
        job_id="job-1",
    )

    assert attached.status == "queued"
    assert attached.job_id == "job-1"
    assert repo.attach_job(
        task_id="task-1",
        run_id=queued.id,
        job_id="job-1",
    ) == attached
    with pytest.raises(DataAnalysisNotFoundError, match="job not found for task"):
        repo.attach_job(
            task_id="task-1",
            run_id=queued.id,
            job_id="job-2",
        )

    with pytest.raises(DataAnalysisTransitionError, match="job.*running"):
        repo.mark_running(
            task_id="task-1",
            run_id=queued.id,
            job_id="job-1",
        )
    with connect(db_path) as conn:
        conn.execute("UPDATE jobs SET status = 'running' WHERE id = 'job-1'")
    repo.mark_running(task_id="task-1", run_id=queued.id, job_id="job-1")
    with pytest.raises(DataAnalysisTransitionError, match="queued"):
        repo.attach_job(
            task_id="task-1",
            run_id=queued.id,
            job_id="job-1",
        )


def test_job_binding_rejects_wrong_kind_and_terminal_jobs(tmp_path):
    db_path = tmp_path / "app.sqlite"
    _seed_domain(db_path)
    repo = DataAnalysisRepository(db_path)

    _seed_job(
        db_path,
        job_id="wrong-kind",
        task_id="task-1",
        kind="join",
    )
    with pytest.raises(DataAnalysisDataError, match="job kind"):
        repo.create_or_get(_identity(), job_id="wrong-kind")
    with connect(db_path) as conn:
        conn.execute("UPDATE jobs SET status = 'failed' WHERE id = 'wrong-kind'")

    _seed_job(
        db_path,
        job_id="terminal",
        task_id="task-1",
        status="failed",
    )
    with pytest.raises(DataAnalysisTransitionError, match="job.*queued"):
        repo.create_or_get(_identity(), job_id="terminal")

    run = repo.create_or_get(_identity())
    _seed_job(
        db_path,
        job_id="wrong-attach",
        task_id="task-1",
        kind="metrics",
    )
    with pytest.raises(DataAnalysisDataError, match="job kind"):
        repo.attach_job(
            task_id="task-1",
            run_id=run.id,
            job_id="wrong-attach",
        )


def test_mark_running_accepts_active_job_transition_but_rejects_terminal_job(
    tmp_path,
):
    db_path = tmp_path / "app.sqlite"
    _seed_domain(db_path)
    repo = DataAnalysisRepository(db_path)
    run = repo.create_or_get(_identity())
    _seed_job(
        db_path,
        job_id="running-job",
        task_id="task-1",
        status="running",
    )

    running = repo.mark_running(
        task_id="task-1",
        run_id=run.id,
        job_id="running-job",
    )
    assert running.status == "running"

    other_run = repo.create_or_get(_identity(config={"histogram_bins": 21}))
    with connect(db_path) as conn:
        conn.execute("UPDATE jobs SET status = 'failed' WHERE id = 'running-job'")
    with pytest.raises(DataAnalysisTransitionError, match="job.*running"):
        repo.mark_running(
            task_id="task-1",
            run_id=other_run.id,
            job_id="running-job",
        )


def test_concurrent_create_or_get_persists_one_row(tmp_path):
    db_path = tmp_path / "app.sqlite"
    _seed_domain(db_path)
    barrier = Barrier(2)

    def create():
        barrier.wait(timeout=5)
        return DataAnalysisRepository(db_path).create_or_get(_identity())

    with ThreadPoolExecutor(max_workers=2) as executor:
        records = list(executor.map(lambda _: create(), range(2)))

    assert records[0] == records[1]
    with connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM data_analysis_runs").fetchone()[0]
    assert count == 1


def test_running_failure_and_explicit_retry_are_cas_guarded(tmp_path):
    db_path = tmp_path / "app.sqlite"
    _seed_domain(db_path)
    _seed_job(db_path, job_id="job-1", task_id="task-1")
    repo = DataAnalysisRepository(db_path)
    queued = repo.create_or_get(_identity(), job_id="job-1")
    with connect(db_path) as conn:
        conn.execute("UPDATE jobs SET status = 'running' WHERE id = 'job-1'")

    running = repo.mark_running(task_id="task-1", run_id=queued.id)
    assert running.status == "running"
    assert running.started_at is not None
    with pytest.raises(DataAnalysisTransitionError, match="queued"):
        repo.mark_running(task_id="task-1", run_id=queued.id)

    failed = repo.fail(
        task_id="task-1",
        run_id=queued.id,
        error_kind="analysis_failed",
        error_message="bad input",
    )
    assert failed.status == "failed"
    assert failed.error_kind == "analysis_failed"
    assert failed.completed_at is not None
    assert repo.create_or_get(_identity()) == failed

    with connect(db_path) as conn:
        conn.execute("UPDATE jobs SET status = 'failed' WHERE id = 'job-1'")
    _seed_job(db_path, job_id="job-2", task_id="task-1")
    retried = repo.retry(_identity(workspace_revision=3), job_id="job-2")
    assert retried.id == queued.id
    assert retried.status == "queued"
    assert retried.job_id == "job-2"
    assert retried.started_at is None
    assert retried.completed_at is None
    assert retried.error_kind is None


def test_cancel_can_retry_but_succeeded_run_cannot(tmp_path):
    db_path = tmp_path / "app.sqlite"
    _seed_domain(db_path)
    repo = DataAnalysisRepository(db_path)
    queued = repo.create_or_get(_identity())
    cancelled = repo.cancel(
        task_id="task-1",
        run_id=queued.id,
        error_kind="user_cancelled",
        error_message="cancelled by user",
    )
    assert cancelled.status == "cancelled"
    assert repo.retry(_identity()).status == "queued"

    repo.mark_running(task_id="task-1", run_id=queued.id)
    artifacts = TaskArtifactRepository(db_path)
    with artifacts.transaction() as conn:
        conn.execute("BEGIN IMMEDIATE")
        artifact = artifacts.register_on_connection(
            conn,
            task_id="task-1",
            kind="data_analysis",
            path="outputs/data-analysis.json",
            content_hash=_sha("result"),
            origin_tool=DATA_ANALYSIS_ARTIFACT_ORIGIN_TOOL,
            provenance=_artifact_provenance(queued),
        )
        succeeded = repo.complete_on_connection(
            conn,
            task_id="task-1",
            run_id=queued.id,
            result_artifact_id=artifact["id"],
            result_content_hash=artifact["content_hash"],
        )
    assert succeeded.status == "succeeded"
    with pytest.raises(DataAnalysisTransitionError, match="failed or cancelled"):
        repo.retry(_identity())


def test_complete_on_connection_shares_artifact_transaction_and_verifies_evidence(
    tmp_path,
):
    db_path = tmp_path / "app.sqlite"
    _seed_domain(db_path)
    analysis = DataAnalysisRepository(db_path)
    artifacts = TaskArtifactRepository(db_path)
    run = analysis.create_or_get(_identity())
    analysis.mark_running(task_id="task-1", run_id=run.id)

    with artifacts.transaction() as conn:
        conn.execute("BEGIN IMMEDIATE")
        artifact = artifacts.register_on_connection(
            conn,
            task_id="task-1",
            kind="data_analysis",
            path="outputs/rolled-back.json",
            content_hash=_sha("rolled-back"),
            origin_tool=DATA_ANALYSIS_ARTIFACT_ORIGIN_TOOL,
            provenance=_artifact_provenance(run),
        )
        completed = analysis.complete_on_connection(
            conn,
            task_id="task-1",
            run_id=run.id,
            result_artifact_id=artifact["id"],
            result_content_hash=artifact["content_hash"],
        )
        assert completed.status == "succeeded"
        conn.rollback()

    assert analysis.get_for_task("task-1", run.id).status == "running"
    assert artifacts.list_for_task("task-1") == []

    with artifacts.transaction() as conn:
        conn.execute("BEGIN IMMEDIATE")
        artifact = artifacts.register_on_connection(
            conn,
            task_id="task-1",
            kind="data_analysis",
            path="outputs/data-analysis.json",
            content_hash=_sha("result"),
            origin_tool=DATA_ANALYSIS_ARTIFACT_ORIGIN_TOOL,
            provenance=_artifact_provenance(run),
        )
        with pytest.raises(DataAnalysisDataError, match="content hash"):
            analysis.complete_on_connection(
                conn,
                task_id="task-1",
                run_id=run.id,
                result_artifact_id=artifact["id"],
                result_content_hash=_sha("wrong"),
            )
        completed = analysis.complete_on_connection(
            conn,
            task_id="task-1",
            run_id=run.id,
            result_artifact_id=artifact["id"],
            result_content_hash=artifact["content_hash"],
        )

    assert analysis.get_for_task("task-1", run.id) == completed
    assert completed.result_artifact_id == artifact["id"]
    assert completed.result_content_hash == artifact["content_hash"]


def test_complete_requires_canonical_data_analysis_artifact_provenance(tmp_path):
    db_path = tmp_path / "app.sqlite"
    _seed_domain(db_path)
    analysis = DataAnalysisRepository(db_path)
    artifacts = TaskArtifactRepository(db_path)
    run = analysis.create_or_get(_identity())
    analysis.mark_running(task_id="task-1", run_id=run.id)

    cases = (
        (
            "wrong-kind",
            "report",
            DATA_ANALYSIS_ARTIFACT_ORIGIN_TOOL,
            _artifact_provenance(run),
            "kind",
        ),
        (
            "wrong-origin",
            "data_analysis",
            "data.describe",
            _artifact_provenance(run),
            "origin",
        ),
        (
            "wrong-provenance",
            "data_analysis",
            DATA_ANALYSIS_ARTIFACT_ORIGIN_TOOL,
            {"input_hash": run.input_hash},
            "provenance",
        ),
        (
            "extra-provenance",
            "data_analysis",
            DATA_ANALYSIS_ARTIFACT_ORIGIN_TOOL,
            {**_artifact_provenance(run), "raw_value": "Alice"},
            "provenance",
        ),
    )
    with artifacts.transaction() as conn:
        conn.execute("BEGIN IMMEDIATE")
        for name, kind, origin, provenance, match in cases:
            artifact = artifacts.register_on_connection(
                conn,
                task_id="task-1",
                kind=kind,
                path=f"outputs/{name}.json",
                content_hash=_sha(name),
                origin_tool=origin,
                provenance=provenance,
            )
            with pytest.raises(DataAnalysisDataError, match=match):
                analysis.complete_on_connection(
                    conn,
                    task_id="task-1",
                    run_id=run.id,
                    result_artifact_id=artifact["id"],
                    result_content_hash=artifact["content_hash"],
                )

        valid = artifacts.register_on_connection(
            conn,
            task_id="task-1",
            kind="data_analysis",
            path="outputs/valid.json",
            content_hash=_sha("valid"),
            origin_tool=DATA_ANALYSIS_ARTIFACT_ORIGIN_TOOL,
            provenance=_artifact_provenance(run),
        )
        completed = analysis.complete_on_connection(
            conn,
            task_id="task-1",
            run_id=run.id,
            result_artifact_id=valid["id"],
            result_content_hash=valid["content_hash"],
        )
    assert completed.status == "succeeded"


def test_complete_rejects_bound_task_job_that_is_no_longer_running(tmp_path):
    db_path = tmp_path / "app.sqlite"
    _seed_domain(db_path)
    _seed_job(db_path, job_id="job-1", task_id="task-1")
    analysis = DataAnalysisRepository(db_path)
    artifacts = TaskArtifactRepository(db_path)
    run = analysis.create_or_get(_identity(), job_id="job-1")
    with connect(db_path) as conn:
        conn.execute("UPDATE jobs SET status = 'running' WHERE id = 'job-1'")
    running = analysis.mark_running(
        task_id="task-1",
        run_id=run.id,
        job_id="job-1",
    )
    with connect(db_path) as conn:
        conn.execute("UPDATE jobs SET status = 'failed' WHERE id = 'job-1'")

    with artifacts.transaction() as conn:
        conn.execute("BEGIN IMMEDIATE")
        artifact = artifacts.register_on_connection(
            conn,
            task_id="task-1",
            kind="data_analysis",
            path="outputs/terminal-job.json",
            content_hash=_sha("terminal-job"),
            origin_tool=DATA_ANALYSIS_ARTIFACT_ORIGIN_TOOL,
            provenance=_artifact_provenance(running),
        )
        with pytest.raises(DataAnalysisTransitionError, match="job.*running"):
            analysis.complete_on_connection(
                conn,
                task_id="task-1",
                run_id=run.id,
                result_artifact_id=artifact["id"],
                result_content_hash=artifact["content_hash"],
            )
        conn.rollback()

    assert analysis.get_for_task("task-1", run.id).status == "running"


def test_succeeded_run_evidence_and_status_are_database_immutable(tmp_path):
    db_path = tmp_path / "app.sqlite"
    _seed_domain(db_path)
    analysis = DataAnalysisRepository(db_path)
    artifacts = TaskArtifactRepository(db_path)
    run = analysis.create_or_get(_identity())
    analysis.mark_running(task_id="task-1", run_id=run.id)
    with artifacts.transaction() as conn:
        conn.execute("BEGIN IMMEDIATE")
        artifact = artifacts.register_on_connection(
            conn,
            task_id="task-1",
            kind="data_analysis",
            path="outputs/immutable.json",
            content_hash=_sha("immutable"),
            origin_tool=DATA_ANALYSIS_ARTIFACT_ORIGIN_TOOL,
            provenance=_artifact_provenance(run),
        )
        completed = analysis.complete_on_connection(
            conn,
            task_id="task-1",
            run_id=run.id,
            result_artifact_id=artifact["id"],
            result_content_hash=artifact["content_hash"],
        )

    with connect(db_path) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="succeeded.*immutable"):
            conn.execute(
                "UPDATE data_analysis_runs SET result_content_hash = ? WHERE id = ?",
                (_sha("replacement"), completed.id),
            )
        with pytest.raises(sqlite3.IntegrityError, match="succeeded.*immutable"):
            conn.execute(
                "UPDATE data_analysis_runs SET job_id = 'replacement-job' WHERE id = ?",
                (completed.id,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="succeeded.*immutable"):
            conn.execute(
                """
                UPDATE data_analysis_runs
                   SET status = 'failed', result_artifact_id = NULL,
                       result_content_hash = NULL, error_kind = 'rewritten',
                       error_message = 'rewritten'
                 WHERE id = ?
                """,
                (completed.id,),
            )

    assert analysis.get_for_task("task-1", completed.id) == completed


def test_complete_conceals_cross_task_run_and_artifact(tmp_path):
    db_path = tmp_path / "app.sqlite"
    _seed_domain(db_path)
    analysis = DataAnalysisRepository(db_path)
    artifacts = TaskArtifactRepository(db_path)
    run = analysis.create_or_get(_identity())
    analysis.mark_running(task_id="task-1", run_id=run.id)

    with artifacts.transaction() as conn:
        conn.execute("BEGIN IMMEDIATE")
        other = artifacts.register_on_connection(
            conn,
            task_id="task-2",
            kind="data_analysis",
            path="outputs/other.json",
            content_hash=_sha("other"),
            origin_tool=DATA_ANALYSIS_ARTIFACT_ORIGIN_TOOL,
            provenance=_artifact_provenance(run),
        )
        with pytest.raises(DataAnalysisNotFoundError, match="run not found for task"):
            analysis.complete_on_connection(
                conn,
                task_id="task-2",
                run_id=run.id,
                result_artifact_id=other["id"],
                result_content_hash=other["content_hash"],
            )
        with pytest.raises(DataAnalysisNotFoundError, match="artifact not found for task"):
            analysis.complete_on_connection(
                conn,
                task_id="task-1",
                run_id=run.id,
                result_artifact_id=other["id"],
                result_content_hash=other["content_hash"],
            )


def test_invalid_or_stale_transitions_fail_without_mutation(tmp_path):
    db_path = tmp_path / "app.sqlite"
    _seed_domain(db_path)
    repo = DataAnalysisRepository(db_path)
    run = repo.create_or_get(_identity())

    with pytest.raises(DataAnalysisTransitionError, match="running"):
        with TaskArtifactRepository(db_path).transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
            repo.complete_on_connection(
                conn,
                task_id="task-1",
                run_id=run.id,
                result_artifact_id="missing",
                result_content_hash=_sha("missing"),
            )
    repo.cancel(task_id="task-1", run_id=run.id)
    with pytest.raises(DataAnalysisTransitionError, match="queued or running"):
        repo.fail(
            task_id="task-1",
            run_id=run.id,
            error_kind="late_failure",
            error_message="late callback",
        )
    with pytest.raises(DataAnalysisNotFoundError, match="run not found for task"):
        repo.mark_running(task_id="task-2", run_id=run.id)


def test_identity_columns_are_immutable_and_corrupt_rows_fail_closed(tmp_path):
    db_path = tmp_path / "app.sqlite"
    _seed_domain(db_path)
    repo = DataAnalysisRepository(db_path)
    run = repo.create_or_get(_identity())

    with connect(db_path) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="identity is immutable"):
            conn.execute(
                "UPDATE data_analysis_runs SET config_json = '{}' WHERE id = ?",
                (run.id,),
            )
        conn.execute(
            "UPDATE data_analysis_runs SET updated_at = 'not-a-timestamp' WHERE id = ?",
            (run.id,),
        )

    with pytest.raises(DataAnalysisDataError, match="corrupt data analysis run"):
        repo.get_for_task("task-1", run.id)
